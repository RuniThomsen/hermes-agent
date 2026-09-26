---
title: Protected output policies
sidebar_label: Protected output
description: Opt-in final-output admission through a profile-owned plugin policy.
---

# Protected output policies

A profile can require a plugin verdict before a turn's final answer is persisted or
returned for delivery. The contract is disabled by default. Configure it in the
owning profile's `config.yaml`:

```yaml
plugins:
  enabled:
    - output-policy
protected_output:
  policy: output-policy
  timeout_seconds: 1.0
```

`policy` is the plugin ID. `timeout_seconds` must be a finite number greater than
zero and at most 30; the default is 1 second. A missing or disabled plugin,
duplicate evaluator, invalid configuration/verdict, exception, timeout, or exhausted
evaluator capacity suppresses the answer. Omit `protected_output` or set it to
`null` to use ordinary output behavior. Any other value opts into the fail-closed
contract, including malformed values.

## Plugin API

A native plugin registers exactly one evaluator in its `register(ctx)` entry point:

```python
from agent.protected_output import OutputDestination, OutputVerdict


def evaluate(destination: OutputDestination, candidate: str) -> OutputVerdict:
    # Replace this example with the profile's policy. An unrecognized destination
    # should be suppressed rather than inferred from the generated text.
    return OutputVerdict("suppress")


def register(ctx):
    ctx.register_output_policy(evaluate)
```

The immutable destination contains `profile_home`, `session_id`, `platform`,
`destination_id`, and `thread_id`. Core binds it before generation using the current
profile and the agent's session/destination. A SessionDB belonging to another
profile, missing session/destination, or the unsupported `codex_app_server` runtime
blocks generation. The evaluator may be synchronous or asynchronous and returns:

For gateway turns, the routed profile and immutable session/destination are bound
before gateway-owned running-status or inactivity notices can run, including before
the agent holder exists. Those interim notices are withheld in protected mode.
An inactivity timeout still hard-interrupts and reaps the worker, but returns an
empty protected result with a fixed failure reason. Activity text and tool names
are withheld from delivery and gateway logs. Unprotected warnings and
approval/control notices remain available.

| Verdict | Outcome |
| --- | --- |
| `OutputVerdict("allow")` | Admit the complete candidate unchanged. |
| `OutputVerdict("replace", "approved text")` | Admit exactly the replacement. |
| `OutputVerdict("suppress")` | Admit an empty answer. |

Dictionaries and extra text on `allow`/`suppress` are invalid. Existing output
transforms precede mandatory policy evaluation. Core finalizer additions, including
budget/recovery/interruption explanations and verifier footers, are evaluated in
the complete candidate. Gateway runtime footers, inferred media, and error-text
fallbacks are withheld after admission so they cannot append unevaluated text.

Evaluation uses at most four concurrent workers per profile plugin manager. A
worker that exceeds its deadline cannot publish its later result; it occupies a
slot until it exits. This is a bounded waiting contract, not cancellation of Python
code. A policy must not publish, log, or persist its candidate independently.

## Persistence, hooks, and compatibility

Protected turns drop intermediate display, reasoning, TTS deltas, commentary, and
tool-progress callbacks. Core transcript/trajectory writes are deferred until the
verdict. Settlement retains prior history and the current user input (including
multimodal content, timestamps, platform message ID, and display metadata), then
appends only the admitted assistant answer. Rejected reasoning and raw response
sidecars are excluded. Memory synchronization and context-engine completion receive
copies of this admitted view. Input-time memory and `pre_llm_call` hooks remain active.

Security and governing hooks continue through the ordinary dispatcher, including
`pre_tool_call` block/modify/approve directives, approval notifications,
`pre_gateway_dispatch`, and `pre_verify`. Output protection does not grant tool
permission. Approval/control UI remains available even while generated commentary
is withheld. Governing plugins remain trusted code and must handle their inputs
without independently exposing raw output.

The explicit raw-observer exclusion list is in `hermes_cli/lifecycle.py`:
`post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`,
`api_request_error`, `pre_auxiliary_call`, `post_auxiliary_call`, `on_stream_start`,
`on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_room_member_activity`,
`subagent_stop`, and `on_session_end`. `post_llm_call` is replayed with the admitted
view after settlement. First-party telemetry and raw diagnostic logging are fenced
during generation. Unknown extension hooks are not silently disabled.

This opt-in mode changes conversation behavior:

- Intermediate tool transcripts are omitted from durable history and the returned
  continuation. Tools still execute normally when authorized; output protection
  does not roll back their effects or sandbox plugin/tool-owned I/O.
- Core tool-result spill files and tool-result presentation metadata callbacks are
  withheld. Large tool results remain in memory for the running turn.
- Compaction and background memory/skill review are suspended for protected turns.
  Long conversations can hit context limits earlier. Context selection still sees
  safe prior history and current input, but is withheld once the current turn
  contains unadmitted assistant/tool messages.
- Final answer delivery uses the returned admitted text; delta-driven TTS has no
  protected-turn audio stream. No promise is made about arbitrary third-party
  callbacks, external runtimes, or adapter behavior outside these core paths.

These are deliberate restrictions, not transparent compatibility. Test the target
adapter and policy in an isolated profile before enabling this mode in production.

Configuration/binding, output-transform, SessionDB persistence, and trajectory
failures produce an empty protected failure result with a fixed `failure_reason`
and `agent_persisted=false`. A failed DB write is not reported as persisted even
when its helper returns `False` without raising. The host does not receive the
exception text as a fallback. A `protected_output_audit` log line records only
decision, session, turn, and status after safe scope reset; it has no candidate
content. Successful rewrite and suppression decisions, blocked turns, and failed
settlements are auditable.

The gateway notice/timeout and settlement failure paths are covered by
`tests/gateway/test_protected_output_delivery.py` and
`tests/agent/test_protected_output_contract.py`. They use a simulated adapter and
provider plus local SessionDB; they do not certify live platform delivery,
third-party plugin I/O, or rollback of authorized tool effects.
