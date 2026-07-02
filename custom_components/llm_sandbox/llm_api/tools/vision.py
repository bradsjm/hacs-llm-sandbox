"""Vision-capable LLM tool for live camera/image snapshot capture.

The returned image bytes stay in this standalone LLM tool path and are never
passed into the Monty sandbox or any Monty-visible facade object.
"""

import base64
import io
import logging
from collections.abc import Mapping
from typing import cast, final, override

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.network import get_url
from homeassistant.util.json import JsonObjectType
from PIL import Image

from ...const import (
    DEFAULT_IMAGE_TARGET_WIDTH,
    MAX_IMAGE_ATTACHMENT_BYTES,
    MAX_IMAGE_TARGET_WIDTH,
    MIN_IMAGE_TARGET_WIDTH,
    TOOL_GET_CAMERA_IMAGE,
)
from ...snapshot import build_snapshot
from ...types import TranslationPlaceholders
from ..errors import tool_error_envelope, tool_error_from_exception
from ..prompts import build_get_camera_image_description
from ._support import _require_loaded_entry, _require_loaded_entry_error

_LOGGER = logging.getLogger(__name__)

ENTITY_NOT_VISIBLE = "entity_not_visible"
CAPTURE_FAILED = "capture_failed"
IMAGE_TOO_LARGE = "image_too_large"
UNSUPPORTED_IMAGE_DOMAIN = "unsupported_image_domain"

# Actionable guidance keyed by the recoverable error key. Message/hints are
# surfaced inline to the LLM so a follow-up call can succeed; stable keys stay
# translated in en.json for the human-facing contract. Mirrors the recorder
# tools' guidance contract so every recoverable tool error is self-remedying.
_VISION_GUIDANCE: dict[str, tuple[str, list[str]]] = {
    ENTITY_NOT_VISIBLE: (
        "Only snapshot-visible entities are capturable.",
        [
            "List visible camera/image entities via execute_home_code "
            "(hass.states.async_all()) and retry with a visible entity_id."
        ],
    ),
    UNSUPPORTED_IMAGE_DOMAIN: (
        "Only camera.* and image.* entities can be captured.",
        ["Retry with a camera.* or image.* entity_id from the visible set."],
    ),
    CAPTURE_FAILED: (
        "The live image capture failed.",
        ["Confirm {entity_id} is online and producing frames, then retry."],
    ),
    IMAGE_TOO_LARGE: (
        "The captured frame exceeds the inline image budget after downscaling.",
        ["Retry with a smaller target_width; {entity_id} produced too many bytes."],
    ),
    "invalid_tool_input": (
        "Invalid tool input.",
        ["Check argument names and types; the validation error was: {error}."],
    ),
}


class _SafeHintDict(dict[str, str]):
    """dict that keeps unknown ``{placeholder}`` tokens verbatim instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _error_guidance(key: str, placeholders: Mapping[str, str]) -> tuple[str | None, list[str] | None]:
    """Return (message, fix) for a vision error key, formatting placeholders."""
    entry = _VISION_GUIDANCE.get(key)
    if entry is None:
        return None, None
    message, templates = entry
    values = _SafeHintDict({str(k): str(v) for k, v in placeholders.items()})
    return message, [template.format_map(values) for template in templates]


def _envelope(key: str, placeholders: TranslationPlaceholders) -> JsonObjectType:
    """Build a recoverable vision error envelope with actionable guidance."""
    if key == ENTITY_NOT_VISIBLE:
        entity_id = placeholders.get("entity_id", "the requested image entity")
        return tool_error_envelope(
            key,
            placeholders,
            message=f"Image entity '{entity_id}' is not visible to this LLM tool.",
        )
    message, fix = _error_guidance(key, placeholders)
    return tool_error_envelope(key, placeholders, message=message, fix=fix)


@final
class GetCameraImageTool(llm.Tool):
    """Capture a visible camera/image entity as a multimodal tool result."""

    name = TOOL_GET_CAMERA_IMAGE
    description = build_get_camera_image_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required(
                "image_entity",
                description="Camera or image entity ID to capture.",
            ): cv.entity_id,
            vol.Optional(
                "target_width",
                description="Maximum output image width in pixels.",
                default=DEFAULT_IMAGE_TARGET_WIDTH,
            ): vol.All(vol.Coerce(float), vol.Range(min=MIN_IMAGE_TARGET_WIDTH, max=MAX_IMAGE_TARGET_WIDTH)),
            vol.Optional(
                "question",
                description="Why the image is being fetched, used as the caption.",
            ): str,
        }
    )

    def __init__(self, entry_id: str) -> None:
        """Initialize the vision tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        try:
            data = cast(dict[str, object], self.parameters(tool_input.tool_args))
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            return _envelope(*mapped)

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            return tool_error_envelope(*setup_error)
        entry = _require_loaded_entry(hass, self.entry_id)
        runtime_data = entry.runtime_data
        assert runtime_data is not None
        settings = runtime_data.settings
        # Build a fresh visible snapshot for every live image read.
        snapshot = build_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )

        image_entity = cast(str, data["image_entity"])
        # The fresh snapshot is the authority for whether this live read is allowed.
        if image_entity not in snapshot.states:
            return _envelope(ENTITY_NOT_VISIBLE, {"entity_id": image_entity})

        domain = image_entity.split(".", 1)[0]
        # Only camera/image domains have a supported frame acquisition path.
        if domain not in {"camera", "image"}:
            return _envelope(UNSUPPORTED_IMAGE_DOMAIN, {"entity_id": image_entity})

        target_width = int(cast(float, data["target_width"]))
        caption = cast(str | None, data.get("question"))
        try:
            return await _capture_image_envelope(hass, image_entity, target_width, caption)
        except HomeAssistantError as err:
            mapped = tool_error_from_exception(err)
            if mapped is None or mapped[0] == "HomeAssistantError":
                return _envelope(CAPTURE_FAILED, {"entity_id": image_entity})
            return _envelope(*mapped)
        except Exception as err:  # noqa: BLE001 - vision capture failures are recoverable tool errors
            mapped = tool_error_from_exception(err)
            if mapped is None:
                _LOGGER.debug("Unexpected image capture failure for %s", image_entity, exc_info=err)
                return _envelope(CAPTURE_FAILED, {"entity_id": image_entity, "error": type(err).__name__})
            return _envelope(*mapped)


async def _fetch_frame_bytes(hass: HomeAssistant, entity_id: str, target_width: int) -> tuple[bytes, str]:
    """Fetch one live frame for a supported visible camera/image entity."""
    domain = entity_id.split(".", 1)[0]
    # Camera entities use HA's camera helper so entity integrations own capture details.
    if domain == "camera":
        return await _async_get_camera_image(hass, entity_id, target_width)

    # Image entities expose an entity_picture URL that is fetched with HA's managed session.
    state = hass.states.get(entity_id)
    picture = state.attributes.get("entity_picture") if state is not None else None
    if not picture:
        # Supported domain but no capturable URL; reclassified to capture_failed by the caller.
        raise HomeAssistantError("image entity has no entity_picture")
    # entity_picture may be either a relative HA path (/api/...) or an absolute URL.
    url = picture if picture.startswith(("http://", "https://")) else f"{get_url(hass)}{picture}"
    session = async_get_clientsession(hass)
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.read(), resp.content_type or "image/jpeg"


async def _async_get_camera_image(hass: HomeAssistant, entity_id: str, target_width: int) -> tuple[bytes, str]:
    """Call HA's camera helper lazily so optional camera deps do not load at import time."""
    from homeassistant.components.camera import async_get_image

    image = await async_get_image(hass, entity_id, width=target_width)
    return image.content, image.content_type


def _downscale(frame_data: bytes, target_width: int) -> tuple[bytes, str]:
    """Downscale and normalize image data to JPEG bytes off the event loop."""
    with Image.open(io.BytesIO(frame_data)) as img:
        processed: Image.Image = img.convert("RGB") if img.mode in ("RGBA", "P", "LA") else img.copy()
        width, height = processed.size
        if width > target_width:
            new_height = max(1, int(target_width * height / width))
            processed = processed.resize((target_width, new_height))
        buffer = io.BytesIO()
        processed.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), "image/jpeg"


async def _capture_image_envelope(
    hass: HomeAssistant,
    entity_id: str,
    target_width: int,
    caption: str | None,
) -> JsonObjectType:
    """Capture, downscale, budget-check, and wrap one inline image attachment."""
    frame_data, _mime = await _fetch_frame_bytes(hass, entity_id, target_width)
    # PIL work is CPU-bound and the resulting bytes stay outside Monty.
    scaled, scaled_mime = await hass.async_add_executor_job(_downscale, frame_data, target_width)
    if len(scaled) > MAX_IMAGE_ATTACHMENT_BYTES:
        return _envelope(IMAGE_TOO_LARGE, {"entity_id": entity_id})
    return cast(
        JsonObjectType,
        {
            "_type": "ha_multimodal_tool_result",
            "text": caption or f"Captured {entity_id}.",
            "attachments": [
                {
                    "kind": "inline_image",
                    "mime_type": scaled_mime,
                    "base64": base64.b64encode(scaled).decode(),
                }
            ],
        },
    )
