# Long repository context in real OpenHands sessions

This is an explicitly modified input protocol for SWE-bench Verified. The two
registered Django issues retain their original issue text, instance IDs, base
commit, tools, model-driven execution and official scoring. Supplying additional
repository context does **not** make this a default OpenHands/SWE-bench result.

The registered pair is `django__django-15268` and `django__django-15278`, both at
`0ab58c120939093fea90822f376e1866fc714d1f`. They are a new cohort outside the earlier
20-task selection. `context_builder.py` records why this is the unique same-repo,
same-base pair in the already pinned 500-row population. No model outcomes or
gold patches select the pair or its shared materials.

## Actual injection point

The pinned official `benchmarks/swebench/run_infer.py` creates `Conversation` in
`SWEBenchEvaluation.evaluate_instance`, sends the original task instruction, and
calls `run_conversation_with_fake_user_response`. `framework_long.install` wraps
only that module's Conversation factory and the instance boundary. Its first
`send_message` becomes:

```python
Message(role="user", content=[
    TextContent(text=original_official_instruction),
    *[TextContent(text=complete_file_block) for complete_file_block in shared_files],
])
```

The original instruction remains the first block without text changes. Each
following block is one complete natural repository file, with its source header.
There is no file repetition or padding. The actual remote workspace is checked
against both the registered base Git blob and current file bytes before sending.
The adapter neither executes repository source nor writes files into the agent
workspace. The official run loop, tools, TaskToolSet delegation, fake-user
follow-ups, patch extraction and scorer stay in place. Delegated subagents still
receive the prompts the model chooses; this experiment does not force the large
context into every subagent.

The SDK RemoteConversation API accepts a `Message` and serializes its individual
content blocks. With the already enabled native tool-calling serializer these
remain list blocks on the OpenAI wire. The checkpoint codec maps each user block
separately. This source-level path still requires an actual SDK wire capture for
the new cohort: verify complete request length, unchanged issue, exact block
token IDs, cross-task prefix difference, selected-row capacity, and current S2
certificate/bank capacity. A context-file check alone cannot certify those.

## Session and cache scope

Run fresh workspaces and independent Conversations A, B, and optionally A again.
Within each arm keep the same service process alive across tasks so its cache can
be reused. Full still creates native sequences with no cached prefix; Prefix(copy)
keeps only real prefix snapshots; S2 keeps its exact-token document bank while
request-local state resets normally. Every arm receives the identical registered
materials. The main nonprefix observation is B after A: task-specific instructions
precede the shared files. A returning after A/B may have a complete historical
prefix match and must be reported as that separate control, not another nonprefix
example. S2's source kind will be `user_text_block`; file identity comes from the
first-message receipt and actual codec offsets.

`first-message.pre-send.json`, `first-message.sent.json`, the exact original
instruction, and `evaluation-returned.json` bind the task, Conversation ID, context
seal, source verification and delivery boundary. The evaluation boundary excludes
container preparation and scoring. Existing service records do not carry official
Conversation IDs: per-task windows and exact initial input hashes must bind those
records externally. Retain auxiliary automatic-title calls in total service cost
and wall time, but classify them separately from agent reasoning/tool turns.

## Integration and remaining admission

The adapter is usable by a new launcher or the actual CPU full-path capture:

```python
handle = framework_long.install(
    run_infer, context_manifest, selection_manifest, receipt_dir)
try:
    run_infer.main()  # original live agent entry, using the new pinned task file
finally:
    framework_long.restore(run_infer, handle)
```

It deliberately does not start services or GPU work. Its standalone local check:

```sh
python framework_long.py --check-only \
  --context-manifest context/context-manifest.json \
  --selection-manifest data/selection-manifest.json
```

The old frozen wrapper hardcodes `tasks20.jsonl`; its patch-baseline and offline
grader admissions also hardcode the earlier Sphinx instance/environment. They
cannot be reused as Django admission. A new launcher needs the newly pinned
two-task file, each official image and environment-baseline proof, the existing
single-attempt greedy policy, and the original official scorer with appropriately
bound new-task image provenance. Do not filter model patches by filename. Any
environment-relative extraction must preserve the raw official patch and all
legitimate agent changes as before.

Before any new GPU launch, obtain the exact base snapshot, create and inspect the
whole-file context, capture both real SDK initial payloads in a CPU sandbox, and
check the declared 95K–110K full-input band and S2 geometry. Fresh GPU numerical
qualification remains required in the eventual independent stage. Current source
and CPU checks are not an agent completion, a performance result, or an accuracy
score. Actual agent wall time and official outcomes compare complete runs;
different model-generated trajectories must not be presented as a same-input
algorithm speedup.
