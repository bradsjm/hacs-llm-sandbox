"""Behavior tests for the get_camera_image LLM tool."""

import base64
import io
from collections.abc import Mapping
from typing import cast
from unittest.mock import patch

import pytest
from custom_components.llm_sandbox.const import TOOL_GET_CAMERA_IMAGE
from custom_components.llm_sandbox.llm_api.tools import vision
from custom_components.llm_sandbox.llm_api.tools.vision import GetCameraImageTool
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType
from PIL import Image as PILImage
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_get_camera_image_returns_multimodal_envelope(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A visible camera returns one strict-base64 inline image attachment."""
    _seed_camera(hass)
    with patch.object(vision, "_async_get_camera_image", return_value=(_tiny_jpeg(), "image/jpeg")):
        result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert result["_type"] == "ha_multimodal_tool_result"
    attachments = cast(list[dict[str, str]], result["attachments"])
    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment["kind"] == "inline_image"
    assert attachment["mime_type"] == "image/jpeg"
    decoded = base64.b64decode(attachment["base64"], validate=True)
    assert decoded


async def test_get_camera_image_rejects_non_visible_entity(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Hidden cameras are rejected by the fresh snapshot visibility check."""
    _seed_camera(hass)
    _seed_camera(hass, object_id="back_yard", name="Back Yard")
    _seed_image(hass, object_id="porch_snapshot", name="Porch Snapshot")
    er.async_get(hass).async_update_entity("camera.front_door", hidden_by=er.RegistryEntryHider.USER)

    result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "entity_not_visible"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]
    candidates = _guidance_candidate_ids(result["error"]["guidance"])
    assert "camera.back_yard" in candidates


async def test_get_camera_image_visibility_is_fresh_per_call(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The vision tool rebuilds visibility on each live image request."""
    _seed_camera(hass)
    with patch.object(vision, "_async_get_camera_image", return_value=(_tiny_jpeg(), "image/jpeg")):
        first = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})
        er.async_get(hass).async_update_entity("camera.front_door", hidden_by=er.RegistryEntryHider.USER)
        second = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert first["_type"] == "ha_multimodal_tool_result"
    assert second["status"] == "error"
    assert second["error"]["key"] == "entity_not_visible"


async def test_get_camera_image_offers_near_miss_candidate(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A near-miss camera ID offers the visible token-matched candidate."""
    _seed_camera(hass)

    result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.frontdoor"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "entity_not_visible"
    assert "camera.front_door" in _guidance_candidate_ids(result["error"]["guidance"])


async def test_get_camera_image_rejects_oversized_capture(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture exceeding the inline attachment budget returns image_too_large."""
    _seed_camera(hass)
    # Lower the budget so a real downscaled frame exceeds it; only the camera
    # fetch seam is mocked, so the real downscale + budget-check path is exercised.
    monkeypatch.setattr(vision, "MAX_IMAGE_ATTACHMENT_BYTES", 1)
    with patch.object(vision, "_async_get_camera_image", return_value=(_tiny_jpeg(), "image/jpeg")):
        result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "image_too_large"
    message = str(result["error"]["message"])
    assert "camera.front_door" in message
    assert str(vision.DEFAULT_IMAGE_TARGET_WIDTH) in message
    assert "1" in message
    assert "target_width" in message


async def test_get_camera_image_maps_capture_failure(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Camera helper failures return the stable capture_failed key."""
    _seed_camera(hass)
    with patch.object(vision, "_async_get_camera_image", side_effect=HomeAssistantError):
        result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "capture_failed"
    assert "camera.front_door" in str(result["error"]["message"])


async def test_get_camera_image_rejects_invalid_tool_input(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Malformed entity IDs are reported as invalid tool input."""
    result = await _call_tool(hass, loaded_entry, {"image_entity": "not_an_entity"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    message = str(result["error"]["message"])
    assert "not_an_entity" in message
    assert "schema" in message


async def test_get_camera_image_rejects_unsupported_visible_domain(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Visible non-camera/image entities return unsupported_image_domain."""
    _seed_camera(hass)
    hass.states.async_set("sensor.foo", "42", {"friendly_name": "Foo"})

    result = await _call_tool(hass, loaded_entry, {"image_entity": "sensor.foo"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "unsupported_image_domain"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]
    assert "sensor.foo" in _guidance_candidate_ids(result["error"]["guidance"])


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"image_entity": "camera.front_door", "target_width": None}, id="null-target-width"),
        pytest.param({"image_entity": "camera.front_door", "target_width": ""}, id="empty-target-width"),
        pytest.param({"image_entity": "camera.front_door", "question": None}, id="null-question"),
        pytest.param({"image_entity": "camera.front_door", "question": ""}, id="empty-question"),
    ],
)
async def test_get_camera_image_omits_empty_optional_args(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> None:
    """Empty/null optional target_width and question are ignored as if omitted."""
    _seed_camera(hass)
    with patch.object(vision, "_async_get_camera_image", return_value=(_tiny_jpeg(), "image/jpeg")):
        result = await _call_tool(hass, loaded_entry, tool_args)

    assert result["_type"] == "ha_multimodal_tool_result"


def _seed_camera(hass: HomeAssistant, *, object_id: str = "front_door", name: str = "Front Door") -> None:
    """Register a visible camera entity and set its live state."""
    er.async_get(hass).async_get_or_create("camera", "test", object_id, suggested_object_id=object_id)
    hass.states.async_set(f"camera.{object_id}", "idle", {"friendly_name": name})


def _seed_image(hass: HomeAssistant, *, object_id: str, name: str) -> None:
    """Register a visible image entity and set its live state."""
    er.async_get(hass).async_get_or_create("image", "test", object_id, suggested_object_id=object_id)
    hass.states.async_set(f"image.{object_id}", "idle", {"entity_picture": "/api/image/test", "friendly_name": name})


def _tiny_jpeg() -> bytes:
    """Build a tiny synthetic JPEG for deterministic capture tests."""
    image = PILImage.new("RGB", (2, 2), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _guidance_candidate_ids(guidance: object) -> set[str]:
    """Return candidate ids from a serialized vision-error guidance payload."""
    assert isinstance(guidance, Mapping)
    candidates = guidance["candidates"]
    assert isinstance(candidates, list)
    return {str(candidate["id"]) for candidate in candidates if isinstance(candidate, Mapping)}


async def _call_tool(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Invoke get_camera_image through its public tool interface."""
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    tool = GetCameraImageTool(entry.entry_id)
    return await tool.async_call(
        hass,
        llm.ToolInput(tool_name=TOOL_GET_CAMERA_IMAGE, tool_args=tool_args),
        llm_context,
    )
