"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

# Only raw-output observers are withheld. Governing hooks (including unknown
# extension hooks) must still traverse the real dispatcher and keep their verdicts.
_PROTECTED_RAW_OBSERVER_HOOKS = frozenset({
    "post_tool_call", "post_llm_call", "pre_api_request", "post_api_request",
    "api_request_error", "pre_auxiliary_call", "post_auxiliary_call",
    "on_stream_start", "on_stream_delta", "on_stream_end", "on_interim_message",
    "on_room_member_activity", "subagent_stop", "on_session_end",
})


def _observe(hook_name: str, **kwargs: Any) -> None:
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)


def _plugin_hooks(hook_name: str, **kwargs: Any) -> List[Any]:
    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    from agent.protected_output import protected_turn
    protected = protected_turn()
    if protected and hook_name in _PROTECTED_RAW_OBSERVER_HOOKS:
        return []
    if not protected:
        _observe(hook_name, **kwargs)
    return _plugin_hooks(hook_name, **kwargs)


async def ainvoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """:func:`invoke_hook` for callers on an event loop: same observers-then-plugins
    composition, with ``async def`` plugin callbacks awaited on that loop."""
    from agent.protected_output import protected_turn
    protected = protected_turn()
    if protected and hook_name in _PROTECTED_RAW_OBSERVER_HOOKS:
        return []
    if not protected:
        _observe(hook_name, **kwargs)
    from hermes_cli import plugins

    return await plugins.ainvoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    _observe("on_session_finalize", **kwargs)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from agent import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    return _plugin_hooks("on_session_finalize", **kwargs)
