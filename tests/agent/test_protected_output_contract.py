"""Protected output must cross a policy before any externally observable write."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def protected_agent(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'model': {'context_length': 65536},
        'protected_output': {'policy': 'test-policy', 'timeout_seconds': 0.05},
    }))
    db = SessionDB(db_path=home / 'state.db')
    with patch('model_tools.get_tool_definitions', return_value=[]), \
         patch('model_tools.check_toolset_requirements', return_value={}), \
         patch('agent.process_bootstrap.OpenAI'):
        agent = AIAgent(
            api_key='test-key-1234567890', base_url='https://example.invalid/v1', model='fake/model',
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            platform='test', chat_id='test-destination', session_id='test-session', session_db=db,
        )
    agent.client = MagicMock()
    yield agent, db, home
    db.close()


def test_missing_required_plugin_withholds_every_output_before_persistence(protected_agent):
    agent, db, home = protected_agent
    delivered = []
    agent.stream_delta_callback = lambda text: delivered.append(('display', text))
    agent.reasoning_callback = lambda text: delivered.append(('reasoning', text))
    agent.interim_assistant_callback = lambda text, **kw: delivered.append(('interim', text))

    def completion(**kwargs):
        agent._fire_stream_delta('RAW candidate')
        agent._fire_reasoning_delta('RAW reasoning')
        agent._emit_interim_assistant_message({'role': 'assistant', 'content': 'RAW interim'})
        agent._session_messages.append({'role': 'assistant', 'content': 'RAW partial'})
        agent._flush_messages_to_session_db(agent._session_messages)
        assert not any('RAW' in str(row) for row in db.get_messages(agent.session_id))
        agent._session_messages.pop()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='RAW candidate', tool_calls=None, reasoning=None), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model='fake/model',
        )

    agent.client.chat.completions.create = completion
    result = agent.run_conversation('hello', stream_callback=lambda text: delivered.append(('tts', text)))
    assert delivered == []
    assert result['final_response'] == ''
    assert result['pre_transform_response'] is None
    assert 'RAW' not in str(result)
    assert 'RAW' not in str(db.get_messages(agent.session_id))


def install_policy(home, mode='replace'):
    plugin = home / 'plugins' / 'test-policy'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text("name: test-policy\nversion: 1.0.0\n")
    (plugin / '__init__.py').write_text('''
from agent.protected_output import OutputVerdict
from dataclasses import FrozenInstanceError
from threading import Event
seen = []
observed = []
released = Event()
finished = Event()

def register(ctx):
    def evaluate(destination, candidate):
        seen.append((destination, candidate))
        try:
            destination.session_id = 'changed'
            raise AssertionError('mutable destination')
        except FrozenInstanceError:
            pass
        mode = MODE
        if mode == 'crash':
            raise RuntimeError('RAW policy error')
        if mode == 'timeout':
            released.wait(5)
            finished.set()
            return OutputVerdict('allow')
        if mode == 'invalid':
            return {'action': 'allow'}
        if mode == 'invalid_text':
            return OutputVerdict('allow', 'unexpected')
        if mode == 'allow':
            return OutputVerdict('allow')
        if mode == 'suppress':
            return OutputVerdict('suppress')
        return OutputVerdict('replace', 'APPROVED')
    ctx.register_output_policy(evaluate)
    ctx.register_hook('transform_llm_output', lambda response_text, **kw: 'TRANSFORM:' + response_text)
    ctx.register_hook('post_llm_call', lambda **kw: observed.append(kw))
'''.replace('MODE', repr(mode)))
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'plugins': {'enabled': ['test-policy']},
        'model': {'context_length': 65536},
        'protected_output': {'policy': 'test-policy', 'timeout_seconds': 0.05},
    }))
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    discover_plugins(force=True)
    return get_plugin_manager()._plugins['test-policy'].module


def complete(agent, content='RAW candidate', interrupt=False):
    def create(**kwargs):
        agent._fire_stream_delta(content)
        agent._fire_reasoning_delta('RAW reasoning')
        if interrupt:
            agent._interrupt_requested = True
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None, reasoning='RAW reasoning'), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model='fake/model',
        )
    agent.client.chat.completions.create = create


@pytest.mark.parametrize('mode,expected', [
    ('allow', 'TRANSFORM:RAW candidate'), ('replace', 'APPROVED'), ('suppress', ''),
    ('crash', ''), ('timeout', ''), ('invalid', ''), ('invalid_text', ''),
])
def test_policy_verdict_is_only_persisted_and_observed_output(protected_agent, mode, expected):
    agent, db, home = protected_agent
    plugin = install_policy(home, mode)
    complete(agent)
    result = agent.run_conversation('hello')
    assert result['final_response'] == expected
    assert len(plugin.seen) == 1
    assert plugin.seen[0][1] == 'TRANSFORM:RAW candidate'
    assert plugin.seen[0][0].profile_home == str(home.resolve())
    assert [r['content'] for r in db.get_messages(agent.session_id) if r['role'] == 'assistant'] == [expected]
    assert result['last_reasoning'] is None
    assert result['pre_transform_response'] is None
    assert len(plugin.observed) == 1
    assert plugin.observed[0]['assistant_response'] == expected
    assert 'RAW reasoning' not in str(plugin.observed)
    if mode == 'timeout':
        plugin.released.set()
        assert plugin.finished.wait(2)
        assert result['final_response'] == ''
        assert 'RAW' not in str(db.get_messages(agent.session_id))


def test_unknown_destination_and_interruption_are_gated(protected_agent):
    agent, db, home = protected_agent
    plugin = install_policy(home, 'allow')
    agent._chat_id = None
    complete(agent, interrupt=True)
    result = agent.run_conversation('hello')
    assert result['final_response'] == ''
    assert plugin.seen == []
    assert 'RAW' not in str(db.get_messages(agent.session_id))


def test_raw_diagnostics_and_managed_observers_are_fenced(protected_agent, caplog, monkeypatch):
    import logging
    from agent.protected_output import output_scope
    from agent import relay_llm, relay_runtime
    agent, db, home = protected_agent
    with output_scope(agent):
        logging.getLogger('agent.test').warning('RAW diagnostic %s', 'RAW argument')
        assert 'RAW' not in caplog.text
        assert agent._dump_api_request_debug({'messages': [{'role': 'assistant', 'content': 'RAW'}]}, reason='test') is None
        assert relay_runtime.relay_instrumentation_enabled() is False
        # A protected attempt must run directly, without feeding request/response to Relay observers.
        called = []
        monkeypatch.setattr(relay_runtime, 'resolve_execution_context', lambda *a: called.append(a))
        assert relay_llm.execute({}, lambda request: 'RAW result', session_id=agent.session_id,
                                 name='test', model_name='fake/model') == 'RAW result'
    assert called == []


@pytest.mark.parametrize('failure', ['db', 'trajectory', 'transform'])
def test_settlement_failures_do_not_escape_or_claim_persistence(protected_agent, monkeypatch, caplog, failure):
    import logging
    from agent import protected_output
    agent, db, home = protected_agent
    install_policy(home, 'allow')
    complete(agent)
    def fail(*args, **kwargs):
        raise RuntimeError('RAW settlement diagnostic')
    if failure == 'db':
        monkeypatch.setattr(agent, '_persist_session', fail)
    elif failure == 'trajectory':
        monkeypatch.setattr(agent, '_save_trajectory', fail)
    else:
        monkeypatch.setattr('agent.turn_finalizer.apply_llm_output_transform', fail)
    with caplog.at_level(logging.INFO):
        result = agent.run_conversation('hello')
    assert result['final_response'] == ''
    assert result['protected_output'] == 'suppress'
    assert result['failed'] is True
    assert result['failure_reason'] == 'protected_output_settlement_failed'
    assert result['agent_persisted'] is False
    assert 'RAW' not in str(result)
    assert 'RAW' not in caplog.text
    assert 'protected_output_audit' in caplog.text
    assert 'status=settlement_failed' in caplog.text


def test_silent_db_write_failure_is_not_reported_as_persisted(protected_agent, monkeypatch):
    agent, db, home = protected_agent
    install_policy(home, 'allow')
    complete(agent)
    monkeypatch.setattr(agent, '_flush_messages_to_session_db', lambda *args: False)
    result = agent.run_conversation('hello')
    assert result['protected_output'] == 'suppress'
    assert result['agent_persisted'] is False
    assert result['failure_reason'] == 'protected_output_settlement_failed'
    assert 'RAW' not in str(db.get_messages(agent.session_id))


def test_binding_failure_blocks_provider_before_generation(protected_agent, monkeypatch):
    agent, db, home = protected_agent
    complete(agent)
    monkeypatch.setattr('agent.protected_output.bind_output', lambda agent: (_ for _ in ()).throw(RuntimeError('RAW binding')))
    result = agent.run_conversation('hello')
    assert result['final_response'] == ''
    assert result['protected_output'] == 'suppress'
    assert result['failed'] is True
    assert result['failure_reason'] == 'protected_output_binding_failed'
    assert 'RAW' not in str(result)


def test_config_load_failure_has_explicit_protected_reason(protected_agent, monkeypatch):
    agent, db, home = protected_agent
    complete(agent)
    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective',
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError('RAW config error')))
    result = agent.run_conversation('hello')
    assert result['final_response'] == ''
    assert result['protected_output'] == 'suppress'
    assert result['failure_reason'] == 'protected_output_binding_failed'
    assert 'RAW' not in str(result)


@pytest.mark.parametrize('mode', ['replace', 'suppress'])
def test_policy_audit_records_decision_without_candidate(protected_agent, caplog, mode):
    import logging
    agent, db, home = protected_agent
    install_policy(home, mode)
    complete(agent)
    with caplog.at_level(logging.INFO):
        agent.run_conversation('hello')
    audit = [record.getMessage() for record in caplog.records
             if record.name == 'agent.protected_output' and 'protected_output_audit' in record.getMessage()]
    assert len(audit) == 1
    assert f'decision={mode}' in audit[0]
    assert 'session=test-session' in audit[0]
    assert 'status=committed' in audit[0]
    assert 'RAW candidate' not in audit[0]


def test_nonstream_reasoning_never_reaches_callback(protected_agent):
    agent, db, home = protected_agent
    delivered = []
    agent.reasoning_callback = delivered.append
    complete(agent, '<think>RAW thought</think>RAW candidate')
    result = agent.run_conversation('hello')
    assert delivered == []
    assert result['last_reasoning'] is None


def test_profile_scope_cannot_rebind_an_existing_agent(protected_agent, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
    agent, db, home = protected_agent
    plugin_a = install_policy(home, 'replace')
    other = tmp_path / 'other'
    other.mkdir()
    token = set_hermes_home_override(str(other))
    try:
        plugin_b = install_policy(other, 'allow')
    finally:
        reset_hermes_home_override(token)
    set_multiplex_active(True)
    history = None
    try:
        for profile, expected in [(home, 'APPROVED'), (other, ''), (home, 'APPROVED')]:
            token = set_hermes_home_override(str(profile))
            secret_token = set_secret_scope({}, profile_home=str(profile))
            try:
                complete(agent)
                result = agent.run_conversation('hello', conversation_history=history)
                assert result['final_response'] == expected
                if profile == home:
                    history = result['messages']
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(token)
        assert len(plugin_a.seen) == 2
        assert not plugin_b.seen
        assert 'RAW' not in str(db.get_messages(agent.session_id))
    finally:
        set_multiplex_active(False)


@pytest.mark.parametrize('scenario', ['recovery', 'interrupted', 'budget', 'footer'])
def test_finalizer_variants_admit_before_trajectory_and_session_writes(protected_agent, monkeypatch, scenario):
    from agent.protected_output import output_scope, settle_output
    from agent.turn_finalizer import finalize_turn
    agent, db, home = protected_agent
    plugin = install_policy(home)
    monkeypatch.chdir(home)
    agent.save_trajectories = True
    agent._current_turn_id = 'test-finalizer'
    agent._file_mutation_verifier_enabled_cache = True
    agent._turn_completion_explainer_enabled_cache = True
    if scenario == 'footer':
        agent._turn_failed_file_mutations = {'test.txt': {'error_preview': 'RAW footer'}}
    messages = [{'role': 'user', 'content': 'hello'},
                {'role': 'assistant', 'content': 'RAW interim', 'reasoning': 'RAW thought'}]
    complete(agent, 'RAW budget summary')
    api_calls = agent.max_iterations if scenario == 'budget' else 1
    with output_scope(agent) as binding:
        raw = finalize_turn(
            agent, final_response=None if scenario == 'budget' else 'RAW candidate',
            api_call_count=api_calls, interrupted=scenario == 'interrupted', failed=False,
            messages=messages, conversation_history=None, effective_task_id='test-task',
            turn_id='test-finalizer', user_message='hello', original_user_message='hello',
            _should_review_memory=False,
            _turn_exit_reason='budget_exhausted' if scenario == 'budget' else 'partial_stream_recovery',
        )
        assert not any(r['role'] == 'assistant' for r in db.get_messages(agent.session_id))
        assert not list(home.glob('*trajector*.jsonl'))
        safe = settle_output(agent, binding, raw, None, 'hello')
    assert safe['final_response'] == 'APPROVED'
    assert 'RAW' not in str(db.get_messages(agent.session_id))
    paths = list(home.glob('*trajector*.jsonl'))
    assert paths
    assert all('RAW' not in p.read_text() and 'APPROVED' in p.read_text() for p in paths)
    candidate = plugin.seen[0][1]
    assert candidate.startswith('TRANSFORM:')
    if scenario == 'footer':
        assert 'RAW footer' in candidate
        assert 'File-mutation verifier' in candidate
    if scenario == 'recovery':
        assert len(candidate) > len('TRANSFORM:RAW candidate')  # completion explanation is evaluated too


def test_unsupported_runtime_never_starts_an_unfenced_provider(protected_agent):
    agent, db, home = protected_agent
    plugin = install_policy(home, 'allow')
    agent.api_mode = 'codex_app_server'
    runtime = MagicMock(return_value={'final_response': 'RAW external runtime', 'messages': []})
    agent._run_codex_app_server_turn = runtime
    result = agent.run_conversation('hello')
    assert not runtime.called
    assert result['final_response'] == ''
    assert 'RAW' not in str(db.get_messages(agent.session_id))


@pytest.mark.parametrize('quiet', [True, False])
def test_tool_round_keeps_commentary_reasoning_and_progress_private(protected_agent, quiet, capsys, monkeypatch):
    agent, db, home = protected_agent
    plugin = install_policy(home)
    seen = []
    from tools.budget_config import BudgetConfig
    monkeypatch.setattr('agent.tool_executor._budget_for_agent', lambda agent: BudgetConfig(
        default_result_size=10, turn_budget=10, preview_size=1, tool_overrides={'todo_list': 10}))
    agent.tool_result_metadata_callback = lambda *args: seen.append(args)
    agent.quiet_mode = quiet
    capsys.readouterr()
    agent.valid_tool_names = {'todo_list'}
    agent.tool_progress_callback = lambda *args: seen.append(args)
    agent.stream_delta_callback = lambda text: seen.append(text)
    agent.reasoning_callback = lambda text: seen.append(text)
    agent.interim_assistant_callback = lambda text, **kwargs: seen.append(text)
    replies = iter([
        SimpleNamespace(content='RAW interim', reasoning=None, tool_calls=[
            SimpleNamespace(id='call-test', type='function', function=SimpleNamespace(
                name='todo_list', arguments='{"todos":[{"id":"t","content":"RAW task","status":"pending"}]}'))]),
        SimpleNamespace(content='RAW final', reasoning=None, tool_calls=None),
    ])
    def completion(**kwargs):
        msg = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason='tool_calls' if msg.tool_calls else 'stop')],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), model='fake/model')
    agent.client.chat.completions.create = completion
    result = agent.run_conversation('hello')
    assert result['final_response'] == 'APPROVED'
    assert agent._todo_store.read()[0]['content'] == 'RAW task'
    assert 'RAW' not in capsys.readouterr().out
    assert not list((home / 'cache' / 'spillover').glob('*'))
    assert seen == []
    assert 'RAW' not in str(db.get_messages(agent.session_id))
    assert 'RAW' not in str(plugin.observed)


def test_unadmitted_transcript_never_reaches_context_engine(protected_agent):
    import logging
    from agent.protected_output import output_scope
    from agent.conversation_loop import _apply_context_engine_selection
    from agent.turn_finalizer import _micro_compact_after_turn
    agent, db, home = protected_agent
    observed = []
    engine = agent.context_compressor
    engine.select_context = lambda messages, **kwargs: observed.append(messages)
    engine._micro_compact_enabled = True
    engine._micro_compact = lambda messages: observed.append(messages) or messages
    messages = [{'role': 'assistant', 'content': 'RAW candidate'}]
    with output_scope(agent):
        assert _apply_context_engine_selection(agent, messages, messages, None, logger=logging.getLogger()) is messages
        _micro_compact_after_turn(agent, messages, 'RAW candidate', logging.getLogger())
        assert observed == []
        # Compression has an independent checkpoint/DB writer; it must stop before
        # entering that pipeline, even when explicitly forced during a protected turn.
        assert agent._compress_context(messages, 'system', force=True) == (messages, 'system')
        safe_input = [{'role': 'user', 'content': 'prior'}, {'role': 'assistant', 'content': 'admitted'},
                      {'role': 'user', 'content': 'current'}]
        agent._persist_user_message_idx = 2
        assert _apply_context_engine_selection(agent, safe_input, safe_input, safe_input[-1],
                                               logger=logging.getLogger()) is safe_input
        assert observed == [safe_input]
    assert not db.get_messages(agent.session_id)


def test_policy_dispatch_resource_failure_suppresses(protected_agent, monkeypatch):
    from agent.protected_output import bind_output
    agent, db, home = protected_agent
    install_policy(home, 'allow')
    binding = bind_output(agent)
    def exhausted(*args, **kwargs):
        raise RuntimeError('thread resources unavailable')
    monkeypatch.setattr('threading.Thread.start', exhausted)
    escaped = False
    try:
        text, decision = binding.evaluate('RAW candidate')
    except RuntimeError:
        escaped = True
    assert not escaped
    assert text == ''


@pytest.mark.parametrize('protect_b', [True, False])
def test_two_profile_policies_keep_cached_history_and_prompt_isolated(protected_agent, tmp_path, protect_b):
    from copy import deepcopy
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
    agent_a, db_a, home_a = protected_agent
    plugin_a = install_policy(home_a, 'replace')
    home_b = tmp_path / 'profile-b'
    home_b.mkdir()
    token = set_hermes_home_override(str(home_b))
    try:
        plugin_b = install_policy(home_b, 'allow')
        if not protect_b:
            config = yaml.safe_load((home_b / 'config.yaml').read_text())
            config.pop('protected_output')
            (home_b / 'config.yaml').write_text(yaml.safe_dump(config))
        db_b = SessionDB(db_path=home_b / 'state.db')
        with patch('model_tools.get_tool_definitions', return_value=[]), \
             patch('model_tools.check_toolset_requirements', return_value={}), \
             patch('agent.process_bootstrap.OpenAI'):
            agent_b = AIAgent(api_key='test-key-1234567890', base_url='https://example.invalid/v1', model='fake/model',
                              quiet_mode=True, skip_context_files=True, skip_memory=True, platform='test',
                              chat_id='test-destination', session_id='test-session', session_db=db_b)
        agent_b.client = MagicMock()
    finally:
        reset_hermes_home_override(token)
    histories = {}
    set_multiplex_active(True)
    try:
        for agent, home, answer in [(agent_a, home_a, 'APPROVED'), (agent_b, home_b, 'TRANSFORM:RAW candidate'),
                                     (agent_a, home_a, 'APPROVED')]:
            token = set_hermes_home_override(str(home))
            secret_token = set_secret_scope({}, profile_home=str(home))
            try:
                old = deepcopy(histories.get(home, []))
                prompt = agent._cached_system_prompt
                complete(agent)
                result = agent.run_conversation('hello', conversation_history=histories.get(home))
                assert result['final_response'] == answer
                assert result['messages'][:len(old)] == old
                if prompt:
                    assert agent._cached_system_prompt == prompt
                histories[home] = result['messages']
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(token)
        assert len(plugin_a.seen) == 2
        assert len(plugin_b.seen) == (1 if protect_b else 0)
        assert 'RAW' not in str(db_a.get_messages(agent_a.session_id))
        assert 'TRANSFORM:RAW candidate' in str(db_b.get_messages(agent_b.session_id))
    finally:
        set_multiplex_active(False)
        db_b.close()


def test_external_memory_receives_only_the_admitted_view(protected_agent):
    agent, db, home = protected_agent
    install_policy(home, 'replace')
    complete(agent)
    provider = agent.client.chat.completions.create
    synced = []
    def completion(**kwargs):
        agent._memory_manager = SimpleNamespace(
            sync_all=lambda *args, **kwargs: synced.append((args, kwargs)),
            queue_prefetch_all=lambda *args, **kwargs: None,
        )
        return provider(**kwargs)
    agent.client.chat.completions.create = completion
    result = agent.run_conversation('hello')
    assert result['final_response'] == 'APPROVED'
    assert len(synced) == 1
    assert synced[0][0][1] == 'APPROVED'
    assert 'RAW' not in str(synced)


def test_governing_hooks_keep_denial_and_approval_across_protection(protected_agent, monkeypatch):
    import asyncio
    from hermes_cli import lifecycle, plugins
    from agent.protected_output import output_scope
    agent, db, home = protected_agent
    install_policy(home)
    manager = plugins.get_plugin_manager()
    seen = []
    directive = {'action': 'block', 'message': 'policy denied'}
    manager._hooks['pre_tool_call'] = [lambda **kw: directive]
    for name in ('pre_approval_request', 'post_approval_response', 'pre_gateway_dispatch', 'pre_verify'):
        manager._hooks[name] = [lambda _name=name, **kw: seen.append(_name) or {'action': 'skip'}]
    def attempt_tool():
        from agent.tool_executor import execute_tool_calls_sequential
        agent.valid_tool_names = {'todo_list'}
        call = SimpleNamespace(id='denied-call', type='function', function=SimpleNamespace(
            name='todo_list', arguments='{"todos":[{"id":"x","content":"must not run","status":"pending"}]}'))
        messages = []
        execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[call]), messages, 'test-task', finalize=False)
        assert 'policy denied' in messages[-1]['content']
        assert not agent._todo_store.has_items()

    with output_scope(agent):
        attempt_tool()
        assert plugins.resolve_pre_tool_block('test_tool', {}) == 'policy denied'
        directive.update(action='approve', message='approval required')
        monkeypatch.setattr('tools.approval.request_tool_approval', lambda *a, **kw: {'approved': False, 'message': 'human denied'})
        assert plugins.resolve_pre_tool_block('test_tool', {}) == 'human denied'
        for name in ('pre_approval_request', 'post_approval_response', 'pre_gateway_dispatch', 'pre_verify'):
            assert lifecycle.invoke_hook(name) == [{'action': 'skip'}]
            assert asyncio.run(lifecycle.ainvoke_hook(name)) == [{'action': 'skip'}]
    assert len(seen) == 8
    directive.update(action='block', message='policy denied')
    agent._protected_output_binding = None
    attempt_tool()
    assert plugins.resolve_pre_tool_block('test_tool', {}) == 'policy denied'


@pytest.mark.parametrize('input_kind', ['multimodal', 'multimodal_display_override', 'prefixed'])
def test_protected_settlement_preserves_input_metadata_and_history(protected_agent, input_kind):
    from copy import deepcopy
    agent, db, home = protected_agent
    install_policy(home)
    complete(agent)
    first = agent.run_conversation('first input')
    history = deepcopy(first['messages'])
    user = [{'type': 'text', 'text': 'describe attachment'},
            {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image.png'}}]
    if input_kind == 'prefixed':
        user = 'API-only prefix: clean input'
    persist_override = None if input_kind == 'multimodal' else 'clean input'
    complete(agent)
    result = agent.run_conversation(user, conversation_history=first['messages'],
                                    persist_user_message=persist_override,
                                    persist_user_timestamp=1234567890.0,
                                    persist_user_platform_id='source-message',
                                    persist_user_display_kind='notification',
                                    persist_user_display_metadata={'source': 'test', 'nested': {'safe': True}})
    assert result['messages'][:len(history)] == history
    row = result['messages'][-2]
    assert row['content'] == ('clean input' if input_kind == 'prefixed' else user)
    assert row['timestamp'] == 1234567890.0
    assert row['platform_message_id'] == 'source-message'
    assert row['display_kind'] == 'notification'
    assert row['display_metadata'] == {'source': 'test', 'nested': {'safe': True}}
    stored = db.get_messages(agent.session_id)
    assert stored[-2]['timestamp'] == row['timestamp']
    assert stored[-2]['display_metadata'] == row['display_metadata']
    assert db.has_platform_message_id(agent.session_id, row['platform_message_id'])
    if input_kind == 'prefixed':
        assert stored[-2]['api_content'] == user
