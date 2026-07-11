"""Prompt profiles, runtime action sections, and tool description builders."""

from .content import (
    build_execute_home_code_description,
    build_get_automation_description,
    build_get_camera_image_description,
    build_get_history_description,
    build_get_logbook_description,
    build_get_statistics_description,
    compose_system_prompt,
    render_home_inventory,
    render_request_location,
    render_tool_capabilities,
)
from .profiles import PROFILE_OPTIONS, PROFILE_REGISTRY, PromptDetail, PromptProfile, resolve_profile

__all__ = [
    "PROFILE_OPTIONS",
    "PROFILE_REGISTRY",
    "PromptDetail",
    "PromptProfile",
    "build_execute_home_code_description",
    "build_get_automation_description",
    "build_get_camera_image_description",
    "build_get_history_description",
    "build_get_logbook_description",
    "build_get_statistics_description",
    "compose_system_prompt",
    "render_home_inventory",
    "render_request_location",
    "render_tool_capabilities",
    "resolve_profile",
]
