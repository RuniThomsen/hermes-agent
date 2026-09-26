"""Gateway must not append unevaluated copy after protected output admission."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.run_turn import GatewayTurnMixin
from gateway.config import Platform
from gateway.turn_context import TurnContext
from gateway.session import SessionSource


@pytest.mark.parametrize('text', ['', 'APPROVED'])
def test_protected_result_survives_gateway_shaping_without_fallback(text, monkeypatch):
    runner = GatewayTurnMixin()
    runner._clear_restart_failure_count = AsyncMock()
    runner.async_session_store = SimpleNamespace(clear_resume_pending=AsyncMock())
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id='test-destination')
    entry = SimpleNamespace(session_id='test-session')
    result = {'final_response': text, 'protected_output': 'replace' if text else 'suppress',
              'failed': True, 'interrupted': True, 'completed': False, 'messages': [], 'session_id': 'test-session'}
    response, silent, messages = asyncio.run(runner._hmwa_shape_agent_response(
        result, source, [], entry, 'test-key', 'test-key', 1, 'test-session', 'test', 0,
    ))
    assert response == text
    assert silent is (not text)
    assert runner._hmwa_classify_turn_failure(result, [], entry) == (False, False, False)
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {
        'display': {'runtime_footer': {'enabled': True, 'fields': ['model']}}})
    result['model'] = 'fake/model'
    assert runner._hmwa_runtime_footer_line(result, source, 1) == ''


@pytest.mark.parametrize('text', ['', 'APPROVED'])
def test_turn_runner_carries_protection_through_delivery(text):
    from unittest.mock import MagicMock
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.session import SessionSource

    class ProtectedAgent:
        def __init__(self, **kwargs):
            self.model = kwargs['model']
            self.session_id = kwargs['session_id']
            self.tools = []
            self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200000)
            self.session_prompt_tokens = self.session_completion_tokens = 0

        def run_conversation(self, message, **kwargs):
            return {'final_response': text, 'protected_output': 'replace' if text else 'suppress',
                    'messages': [], 'failed': False, 'completed': True, 'response_transformed': True}

    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._session_db = None
    runner._prefill_messages = None
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.return_value = ('test-model', {})
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._resolve_turn_agent_config.return_value = {'model': 'test-model', 'runtime': {}}
    runner._agent_config_signature.return_value = ('test-signature',)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    runner._is_telegram_topic_lane.return_value = False
    runner._is_discord_auto_thread_lane.return_value = False
    runner._is_relay_discord_channel_lane.return_value = False
    ctx = TurnContext(source=SessionSource(platform=Platform.LOCAL, chat_id='test-destination', user_id='test-user'),
                      message='hello', history=[], session_id='test-session', session_key='test-key', user_config={},
                      AIAgent=ProtectedAgent, resolve_display_setting=lambda *args: False,
                      _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    result = TurnRunner(runner, ctx).run_sync()
    assert result['final_response'] == text
    assert result['protected_output'] == ('replace' if text else 'suppress')


def test_api_response_items_preserve_policy_silence():
    from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin
    result = {'final_response': '', 'protected_output': 'suppress', 'messages': []}
    items = OpenAICompatRoutesMixin._extract_output_items(result)
    assert items[-1]['content'][0]['text'] == ''


def test_gateway_owned_notices_and_timeout_are_protected_before_agent_exists(tmp_path, monkeypatch, caplog):
    import yaml
    from unittest.mock import MagicMock
    home = tmp_path / 'owner'
    home.mkdir()
    (home / 'config.yaml').write_text(yaml.safe_dump({'protected_output': {'policy': 'missing'}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    source = SessionSource(platform=Platform.LOCAL, chat_id='destination', user_id='user')
    ctx = TurnContext(source=source, session_id='session', session_key='key')
    adapter = SimpleNamespace(send=AsyncMock(), edit_message=AsyncMock(), emit_warning=AsyncMock())
    runner = GatewayTurnMixin()
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._delivery_adapter_for = lambda source: adapter
    runner._run_agent_progress_threading = lambda *args: (None, None, None)
    runner.hooks = SimpleNamespace()
    bridge = MagicMock()
    async def bind():
        runner._run_agent_bind_turn_wiring(ctx, bridge, source, None, False)
    asyncio.run(bind())
    assert ctx.agent_holder[0] is None
    assert ctx._protected_output_binding.destination.session_id == 'session'
    worker = SimpleNamespace(agent_warning=60, agent_timeout=120)
    asyncio.run(runner._run_agent_inactivity_warning(worker, ctx))
    assert adapter.emit_warning.await_count == 0

    class Display:
        user_config = {}
        platform_key = 'local'
        resolve_display_setting = staticmethod(lambda *args: False)
        _display_surface_mode = staticmethod(lambda *args, **kwargs: 'all')
        _generic_status_phrase = staticmethod(lambda *args: 'status')

    async def no_wait(_):
        return None

    monkeypatch.setattr('gateway.run_turn.asyncio.sleep', no_wait)
    runner._should_emit_long_running_notification = lambda *args: True
    task = asyncio.run(runner._run_agent_notify_long_running(Display(), ctx, [object()]))
    assert adapter.send.await_count == 0
    assert adapter.edit_message.await_count == 0
    assert task is None

    worker = SimpleNamespace(agent_timeout=120)
    ctx.agent_holder[0] = SimpleNamespace(get_activity_summary=lambda: {
        'last_activity_desc': 'RAW activity', 'current_tool': 'RAW tool',
        'seconds_since_activity': 120, 'api_call_count': 1, 'max_iterations': 5})
    monkeypatch.setattr('gateway.run.request_hard_interrupt', lambda *args, **kwargs: None)
    result = runner._run_agent_timeout_result(worker, ctx)
    assert result['protected_output'] == 'suppress'
    assert result['final_response'] == ''
    assert result['agent_persisted'] is False
    assert 'activity' not in str(result).lower()
    assert 'RAW' not in caplog.text


def test_unprotected_inactivity_warning_remains_available():
    runner = GatewayTurnMixin()
    source = SessionSource(platform=Platform.LOCAL, chat_id='destination', user_id='user')
    adapter = SimpleNamespace(emit_warning=AsyncMock())
    runner._delivery_adapter_for = lambda source: adapter
    ctx = TurnContext(source=source, _status_thread_metadata=None)
    worker = SimpleNamespace(agent_warning=60, agent_timeout=120)
    asyncio.run(runner._run_agent_inactivity_warning(worker, ctx))
    assert adapter.emit_warning.await_count == 1
    assert '/stop' in adapter.emit_warning.await_args.args[1]


def test_gateway_notice_binding_uses_routed_profile_for_each_turn(tmp_path):
    import yaml
    from unittest.mock import MagicMock
    homes = {}
    for name, protected in [('a', True), ('b', False)]:
        home = tmp_path / name
        home.mkdir()
        (home / 'config.yaml').write_text(yaml.safe_dump(
            {'protected_output': {'policy': 'missing'}} if protected else {}))
        homes[name] = home
    runner = GatewayTurnMixin()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda source: homes[source.chat_id]
    runner._run_agent_progress_threading = lambda *args: (None, None, None)
    runner._delivery_adapter_for = lambda source: None
    runner.hooks = SimpleNamespace()
    bridge = MagicMock()

    async def bind(name):
        source = SessionSource(platform=Platform.LOCAL, chat_id=name, user_id='user')
        ctx = TurnContext(source=source, session_id=f'session-{name}')
        runner._run_agent_bind_turn_wiring(ctx, bridge, source, None, False)
        return ctx._protected_output_binding

    for name, expected in [('a', True), ('b', False), ('a', True)]:
        binding = asyncio.run(bind(name))
        assert (binding is not None) is expected
        if binding is not None:
            assert binding.destination.profile_home == str(homes[name].resolve())
            assert binding.destination.destination_id == name
            assert binding.destination.session_id == f'session-{name}'


def test_gateway_scope_failure_withholds_pre_agent_notices(monkeypatch):
    from unittest.mock import MagicMock
    runner = GatewayTurnMixin()
    source = SessionSource(platform=Platform.LOCAL, chat_id='destination', user_id='user')
    ctx = TurnContext(source=source, session_id='session')
    runner._run_agent_progress_threading = lambda *args: (None, None, None)
    runner._delivery_adapter_for = lambda source: None
    runner._profile_scope_for_source = lambda source: (_ for _ in ()).throw(RuntimeError('scope failed'))
    runner.hooks = SimpleNamespace()

    async def bind():
        runner._run_agent_bind_turn_wiring(ctx, MagicMock(), source, None, False)

    asyncio.run(bind())
    assert ctx._protected_output_binding is not None
    assert ctx._protected_output_binding.owns_profile is False
