"""Behavior tests for the get_camera_image LLM tool."""

import base64
import io
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
    er.async_get(hass).async_update_entity("camera.front_door", hidden_by=er.RegistryEntryHider.USER)

    result = await _call_tool(hass, loaded_entry, {"image_entity": "camera.front_door"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "entity_not_visible"
    assert result["error"]["placeholders"]["entity_id"] == "camera.front_door"
    assert result["error"]["hints"]


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
    assert result["error"]["hints"]


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
    assert result["error"]["hints"]


async def test_get_camera_image_rejects_invalid_tool_input(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Malformed entity IDs are reported as invalid tool input."""
    result = await _call_tool(hass, loaded_entry, {"image_entity": "not_an_entity"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    assert result["error"]["hints"]


async def test_get_camera_image_rejects_unsupported_visible_domain(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Visible non-camera/image entities return unsupported_image_domain."""
    hass.states.async_set("sensor.foo", "42", {"friendly_name": "Foo"})

    result = await _call_tool(hass, loaded_entry, {"image_entity": "sensor.foo"})

    assert result["status"] == "error"
    assert result["error"]["key"] == "unsupported_image_domain"
    assert result["error"]["hints"]


def _seed_camera(hass: HomeAssistant) -> None:
    """Register a visible camera entity and set its live state."""
    er.async_get(hass).async_get_or_create("camera", "test", "front_door", suggested_object_id="front_door")
    hass.states.async_set("camera.front_door", "idle", {"friendly_name": "Front Door"})


def _tiny_jpeg() -> bytes:
    """Build a tiny synthetic JPEG for deterministic capture tests."""
    image = PILImage.new("RGB", (2, 2), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


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
