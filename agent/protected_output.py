"""Opt-in, turn-scoped output admission. Policies never receive a publishing capability.

A protected turn has no intermediate transcript: only the accepted input and admitted
final answer become durable. Provider aggregation owns the candidate; deltas are dropped.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import logging
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from typing import Callable


@dataclass(frozen=True)
class OutputDestination:
    profile_home: str
    session_id: str
    platform: str
    destination_id: str
    thread_id: str


@dataclass(frozen=True)
class OutputVerdict:
    action: str
    text: str | None = None


@dataclass(frozen=True)
class OutputBinding:
    destination: OutputDestination
    policy: Callable | None
    timeout: float
    slots: BoundedSemaphore
    owns_profile: bool = True

    def evaluate(self, candidate: str) -> tuple[str, str]:
        if not self.owns_profile or not self.policy or not self.destination.destination_id or not self.destination.session_id:
            return '', 'unavailable'
        if not self.slots.acquire(blocking=False):
            return '', 'busy'
        result = Queue(maxsize=1)
        policy, destination = self.policy, self.destination

        def evaluate():
            try:
                verdict = policy(destination, candidate)
                if inspect.isawaitable(verdict):
                    verdict = asyncio.run(verdict)
                result.put_nowait(verdict)
            except BaseException:
                result.put_nowait(None)
            finally:
                self.slots.release()

        # Bounded concurrency, copied profile scope, daemon lifetime. A late result can
        # only reach this private queue, never a callback, transcript or future turn.
        try:
            Thread(target=copy_context().run, args=(evaluate,), daemon=True, name='output-policy').start()
        except Exception:
            self.slots.release()
            return '', 'unavailable'
        try:
            verdict = result.get(timeout=self.timeout)
        except Empty:
            return '', 'timeout'
        if type(verdict) is not OutputVerdict:
            return '', 'invalid'
        if verdict.action == 'allow' and verdict.text is None:
            return candidate, 'allow'
        if verdict.action == 'replace' and type(verdict.text) is str:
            return verdict.text, 'replace'
        if verdict.action == 'suppress' and verdict.text is None:
            return '', 'suppress'
        return '', 'invalid'


_active: ContextVar[OutputBinding | None] = ContextVar('protected_output', default=None)


def audit_output(binding: OutputBinding, decision: str, status: str, turn_id: str) -> None:
    """Record only structural outcome after temporarily lifting the log-content fence."""
    token = _active.set(None)
    try:
        logging.getLogger('agent.protected_output').info(
            'protected_output_audit decision=%s session=%s turn=%s status=%s',
            decision, binding.destination.session_id, turn_id, status,
        )
    finally:
        _active.reset(token)


def protected_turn(agent=None) -> bool:
    return _active.get() is not None or isinstance(getattr(agent, '_protected_output_binding', None), OutputBinding)


def unavailable_binding(agent) -> OutputBinding:
    return OutputBinding(OutputDestination('', str(getattr(agent, 'session_id', '') or ''),
        str(getattr(agent, 'platform', '') or ''), str(getattr(agent, '_chat_id', '') or ''),
        str(getattr(agent, '_thread_id', '') or '')), None, 1.0, BoundedSemaphore(1), False)


def bind_output(agent) -> OutputBinding | None:
    from hermes_constants import get_hermes_home
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.plugins import get_plugin_manager

    home = get_hermes_home().resolve()
    # Unreadable configuration cannot prove the profile is unprotected.
    try:
        config = load_user_config_effective(fail_closed=True)
        setting = config.get('protected_output')
    except Exception:
        return unavailable_binding(agent)
    if setting is None:
        return None
    manager = get_plugin_manager()
    policy = None
    timeout = 1.0
    if isinstance(setting, dict):
        name = setting.get('policy')
        value = setting.get('timeout_seconds', timeout)
        if type(value) in (int, float) and math.isfinite(value) and 0 < value <= 30:
            timeout = float(value)
            try:
                manager.discover_and_load()
                callbacks = manager.iter_hook_callbacks('protected_output:' + name) if isinstance(name, str) and name else ()
                if len(callbacks) == 1:
                    policy = callbacks[0]
            except Exception:
                policy = None
    destination = OutputDestination(
        str(home), str(getattr(agent, 'session_id', '') or ''),
        str(getattr(agent, 'platform', '') or ''), str(getattr(agent, '_chat_id', '') or ''),
        str(getattr(agent, '_thread_id', '') or ''),
    )
    from pathlib import Path
    db = getattr(agent, '_session_db', None)
    owns_profile = db is None or Path(db.db_path).resolve().parent == home
    return OutputBinding(destination, policy, timeout, manager._output_policy_slots, owns_profile)


@contextmanager
def output_scope(agent):
    try:
        binding = bind_output(agent)
    except Exception:
        binding = unavailable_binding(agent)
    agent._protected_output_binding = binding
    token = _active.set(binding)
    try:
        yield binding
    finally:
        _active.reset(token)
        # Keep the agent fence after return: late provider writers cannot publish.
        # The next admitted turn replaces it before generation.


def run_output_turn(binding, run, agent, *args, **kwargs):
    if binding is None:
        return run(agent, *args, **kwargs)
    if (not binding.owns_profile or not binding.destination.destination_id
            or not binding.destination.session_id or agent.api_mode == 'codex_app_server'):
        return {'final_response': '', 'failed': True, 'completed': False,
                'protected_output_blocked': True, 'failure_reason': 'protected_output_binding_failed',
                'session_id': binding.destination.session_id}
    try:
        return run(agent, *args, **kwargs)
    except (Exception, KeyboardInterrupt):
        # No exception text or partial SDK response may escape to a host fallback.
        return {'final_response': '', 'failed': True, 'completed': False,
                'protected_output_blocked': True, 'failure_reason': 'protected_output_generation_failed',
                'session_id': binding.destination.session_id}


def settle_output(agent, binding, result, history, user_message, *, stored_user_message=None):
    """One commit point for normal, recovery and early-return envelopes.

    No raw provider sidecars, rejected candidates or intermediate tool transcripts
    escape in the result. Prior turns are retained byte-for-byte.
    """
    from agent.message_metadata import append_message
    from agent.turn_finalizer import apply_llm_output_transform

    candidate = result.get('final_response') or ''
    if not isinstance(candidate, str):
        candidate = ''
    candidate, _, _ = apply_llm_output_transform(
        agent, candidate, turn_id=getattr(agent, '_current_turn_id', '') or '',
    )
    text, decision = ('', 'suppress') if result.get('protected_output_blocked') else binding.evaluate(candidate)
    from copy import deepcopy
    messages = deepcopy(history or [])
    user_row = deepcopy(stored_user_message) if stored_user_message is not None else {'role': 'user', 'content': user_message}
    from agent.session_persistence import _override_replaces_content
    content = user_row.get('content')
    if _override_replaces_content(user_row, content, user_message):
        if isinstance(content, str) and content != user_message and user_row.get('api_content') is None:
            user_row['api_content'] = content
        user_row['content'] = deepcopy(user_message)
    append_message(messages, user_row)
    append_message(messages, {'role': 'assistant', 'content': text})
    # Construct the safe envelope rather than hunting raw fields after each producer
    # adds a new sidecar. Only scalar operational metadata is carried forward.
    safe = {key: value for key, value in result.items() if key in {
        'api_calls', 'completed', 'failed', 'partial', 'interrupted', 'model', 'provider',
        'base_url', 'session_id', 'input_tokens', 'output_tokens', 'cache_read_tokens',
        'cache_write_tokens', 'reasoning_tokens', 'prompt_tokens', 'completion_tokens',
        'total_tokens', 'estimated_cost_usd', 'cost_status', 'cost_source', 'last_prompt_tokens',
        'service_tier',
    } and (value is None or type(value) in (str, bool, int, float))}
    safe.update(final_response=text, messages=messages, last_reasoning=None,
                pre_transform_response=None, response_transformed=True, response_previewed=False,
                protected_output=decision, turn_exit_reason='protected_output:' + decision)
    if result.get('protected_output_blocked'):
        reason = result.get('failure_reason')
        safe['failure_reason'] = reason if reason in {
            'protected_output_binding_failed', 'protected_output_generation_failed',
        } else 'protected_output_generation_failed'
    safe['agent_persisted'] = False
    agent._llm_output_transform = None
    agent._session_messages = messages
    # Publication remains fenced; only the explicit persistence calls below may write
    # the safe transcript. The agent guard remains active against orphan writers.
    agent._protected_output_commit = messages
    try:
        if binding.owns_profile:
            persisted = agent._persist_session(messages, history)
            if getattr(agent, '_session_db', None) is not None and persisted is False:
                raise RuntimeError('Protected output persistence failed')
            agent._save_trajectory(messages, str(user_message), bool(safe.get('completed')))
            safe['agent_persisted'] = getattr(agent, '_session_db', None) is not None
    finally:
        agent._protected_output_commit = None
    # Observers only receive structural copies of the committed view. They have no
    # reference through which to change the verdict or the outbound envelope.
    from copy import deepcopy
    from hermes_cli.lifecycle import invoke_hook
    from agent.conversation_loop import _notify_context_engine_turn_complete, logger
    if not binding.owns_profile:
        audit_output(binding, decision, 'blocked', getattr(agent, '_current_turn_id', '') or '')
        return safe
    token = _active.set(None)
    try:
        invoke_hook('post_llm_call', session_id=binding.destination.session_id,
                    task_id='', turn_id=getattr(agent, '_current_turn_id', '') or '',
                    user_message=user_message, assistant_response=text,
                    conversation_history=deepcopy(messages), model=agent.model,
                    platform=binding.destination.platform)
        _notify_context_engine_turn_complete(agent, deepcopy(messages), logger=logger)
        agent._sync_external_memory_for_turn(
            original_user_message=user_message, final_response=text,
            interrupted=bool(safe.get('interrupted')), messages=deepcopy(messages),
        )
    except Exception:
        logger.warning('Protected output observer failed')
    finally:
        _active.reset(token)
    audit_output(binding, decision, 'committed', getattr(agent, '_current_turn_id', '') or '')
    return safe


def settlement_failure(agent, binding, history, user_message):
    """Empty protected envelope when admission or durable settlement cannot finish."""
    from copy import deepcopy
    from agent.message_metadata import append_message
    messages = deepcopy(history or [])
    append_message(messages, {'role': 'user', 'content': deepcopy(user_message)})
    append_message(messages, {'role': 'assistant', 'content': ''})
    agent._protected_output_commit = None
    agent._llm_output_transform = None
    agent._session_messages = messages
    audit_output(binding, 'suppress', 'settlement_failed', getattr(agent, '_current_turn_id', '') or '')
    return {'final_response': '', 'messages': messages, 'failed': True, 'completed': False,
            'failure_reason': 'protected_output_settlement_failed', 'protected_output': 'suppress',
            'agent_persisted': False, 'response_transformed': True, 'last_reasoning': None,
            'pre_transform_response': None, 'session_id': binding.destination.session_id}


def may_persist(agent, messages) -> bool:
    return not protected_turn(agent) or messages is getattr(agent, '_protected_output_commit', None)
