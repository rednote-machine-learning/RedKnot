"""External, CPU-only adapter for real long-context SWE-bench conversations.

install() changes only the first user Message supplied by the official
evaluate_instance method. The official Agent, tools, run loop, patch extraction
and scorer are untouched. This is supplemental-context research, not the
default SWE-bench input protocol. Importing this module never starts a model.
"""
from __future__ import annotations
import argparse
import base64
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import shlex
import threading
import time

import context_builder as cb

UPSTREAM_INFER_SHA256 = '6cd1e84bf8b6d659993c8ce22372bb1140ffb05bb37f3cf821fd9b7b58e27caa'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _write(path, body):
    value = cb.seal(body)
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return value


def _workspace_program(manifest, repo_path):
    # This is our fixed read-only verifier, never repository-provided code.
    arguments = dict(repo_path=repo_path, base_commit=manifest['base_commit'],
                     files=[{key: item[key] for key in ('path', 'sha256', 'git_blob_sha1')}
                            for item in manifest['files']])
    encoded = base64.b64encode(json.dumps(arguments).encode()).decode()
    return '''import base64,hashlib,json,pathlib,subprocess
a=json.loads(base64.b64decode(ENCODED))
r=pathlib.Path(a['repo_path']).resolve()
def git(*args):
 return subprocess.check_output(['git','-C',str(r),*args],text=True).strip()
assert git('rev-parse','--verify',a['base_commit']+'^{commit}')==a['base_commit'], 'base commit unavailable'
files=[]
for item in a['files']:
 p=r/item['path']
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(r), 'unsafe source path'
 raw=p.read_bytes()
 actual=hashlib.sha256(raw).hexdigest()
 blob=git('rev-parse',a['base_commit']+':'+item['path'])
 assert actual==item['sha256'] and blob==item['git_blob_sha1'], 'workspace/base source differs: '+item['path']
 assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\\0'+raw).hexdigest()==blob, 'working source differs from base blob'
 files.append(dict(path=item['path'],sha256=actual,git_blob_sha1=blob,bytes=len(raw)))
print(json.dumps(dict(repo_path=str(r),head=git('rev-parse','HEAD'),base_commit=a['base_commit'],files=files,task_code_executed=False),sort_keys=True))
'''.replace('ENCODED', repr(encoded))


def verify_actual_workspace(workspace, repo_path, manifest):
    """Check copied working files and Git base blobs before the first model call."""
    command = 'python3 -c ' + shlex.quote(_workspace_program(manifest, repo_path))
    result = workspace.execute_command(command)
    require(result.exit_code == 0, 'shared repository context does not match the actual workspace: ' + str(result.stderr))
    value = json.loads(result.stdout)
    require(value['base_commit'] == manifest['base_commit']
            and len(value['files']) == len(manifest['files'])
            and value['task_code_executed'] is False, 'workspace source receipt is incomplete')
    for expected, actual in zip(manifest['files'], value['files']):
        require(all(actual[key] == expected[key] for key in ('path', 'sha256', 'git_blob_sha1')),
                'workspace source receipt differs from the registered context')
    return value


def check_context(context_manifest, selection_manifest):
    path = Path(context_manifest).resolve()
    manifest = cb.read_sealed(path)
    blocks = cb.load_content_blocks(path)
    selection, rows = cb.validate_selection(selection_manifest)
    require(manifest['selection_sha256'] == selection['sha256']
            and manifest['task_ids'] == selection['selected_instance_ids'], 'context and task selection are detached')
    return manifest, blocks, selection, rows


class _FirstMessageProxy:
    def __init__(self, conversation, installation, state):
        self._conversation = conversation
        self._installation = installation
        self._state = state

    def __getattr__(self, key):
        return getattr(self._conversation, key)

    def send_message(self, message, sender=None):
        state, owner = self._state, self._installation
        if state['delivered']:
            return self._conversation.send_message(message, sender=sender)
        require(isinstance(message, str) and state['row']['problem_statement'] in message,
                'the original official task instruction must be the first message')
        source = verify_actual_workspace(state['workspace'], state['repo_path'], owner.manifest)
        contents = [owner.TextContent(text=message)]
        contents.extend(owner.TextContent(text=block['text']) for block in owner.blocks)
        outbound = owner.Message(role='user', content=contents)
        receipt = _write(state['directory'] / 'first-message.pre-send.json', dict(
            schema='s2-long-first-user-message-v1', instance_id=state['row']['instance_id'],
            conversation_id=str(getattr(self._conversation, 'id', None)),
            context_sha256=owner.manifest['sha256'], selection_sha256=owner.selection['sha256'],
            original_instruction_sha256=hashlib.sha256(message.encode()).hexdigest(),
            original_instruction_preserved=True, original_instruction_first=True,
            message_role='user', content_block_count=len(contents),
            shared_blocks=[dict(content_block_index=i+1, path=item['path'],
                                block_sha256=item['block_sha256'], source_sha256=item['sha256'])
                           for i, item in enumerate(owner.manifest['files'])],
            actual_workspace_source=source, supplemental_context=True,
            official_default_prompt_protocol=False, model_execution_claim=False,
            original_instruction_file='original-instruction.txt',
            event='before_actual_remote_send_message', unix=time.time()))
        with (state['directory'] / 'original-instruction.txt').open('x', encoding='utf-8', newline='') as stream:
            stream.write(message)
        result = self._conversation.send_message(outbound, sender=sender)
        state['delivered'] = True
        state['message_sha256'] = receipt['sha256']
        _write(state['directory'] / 'first-message.sent.json', dict(
            schema='s2-long-first-user-message-sent-v1', first_message_sha256=receipt['sha256'],
            instance_id=state['row']['instance_id'], conversation_id=receipt['conversation_id'],
            event='actual_remote_send_message_returned', unix=time.time()))
        return result


class Installation:
    def __init__(self, module, context_manifest, selection_manifest, receipt_dir, *,
                 expected_source_sha256=UPSTREAM_INFER_SHA256, message_types=None):
        self.manifest, self.blocks, self.selection, rows = check_context(context_manifest, selection_manifest)
        require(cb.file_sha(module.__file__) == expected_source_sha256, 'official inference source changed')
        self.rows = {row['instance_id']: row for row in rows}
        self.module = module
        self.cls = module.SWEBenchEvaluation
        self.original_evaluate = self.cls.evaluate_instance
        self.original_conversation = module.Conversation
        self.directory = Path(receipt_dir).resolve() / 'long-context'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active = ContextVar('s2_long_instance', default=None)
        self.lock, self.ordinal = threading.Lock(), 0
        if message_types is None:
            from openhands.sdk.llm import Message, TextContent
            message_types = Message, TextContent
        self.Message, self.TextContent = message_types

        def conversation(*args, **kwargs):
            state = self.active.get()
            require(state is not None and not state['created'], 'unexpected conversation creation outside an admitted task')
            state['created'] = True
            return _FirstMessageProxy(self.original_conversation(*args, **kwargs), self, state)

        def evaluate(evaluator, instance, workspace):
            row = self.rows.get(instance.id)
            require(row is not None and all(instance.data[key] == row[key]
                    for key in ('instance_id', 'repo', 'base_commit', 'problem_statement')),
                    'unregistered or modified task cannot receive the shared context')
            with self.lock:
                ordinal = self.ordinal
                self.ordinal += 1
            directory = self.directory / f'{ordinal:04d}.{instance.id}'
            directory.mkdir(exist_ok=False)
            state = dict(row=row, workspace=workspace, repo_path=evaluator.get_repo_path(instance),
                         directory=directory, delivered=False, created=False, message_sha256=None)
            token = self.active.set(state)
            started = time.time()
            error = None
            try:
                result = self.original_evaluate(evaluator, instance, workspace)
                require(state['delivered'], 'official evaluation returned without the admitted first message')
                return result
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                self.active.reset(token)
                _write(directory / 'evaluation-returned.json', dict(
                    schema='s2-long-evaluation-boundary-v1', instance_id=instance.id,
                    context_sha256=self.manifest['sha256'], selection_sha256=self.selection['sha256'],
                    first_message_sha256=state['message_sha256'], first_message_delivered=state['delivered'],
                    started_unix=started, ended_unix=time.time(), error=error,
                    boundary='official evaluate_instance only; excludes outer container preparation and grading',
                    task_success_or_official_score_claim=False))

        self.conversation_hook, self.evaluate_hook = conversation, evaluate
        module.Conversation = conversation
        self.cls.evaluate_instance = evaluate

    def close(self):
        require(self.module.Conversation is self.conversation_hook
                and self.cls.evaluate_instance is self.evaluate_hook, 'long adapter hook ownership changed')
        self.module.Conversation = self.original_conversation
        self.cls.evaluate_instance = self.original_evaluate


def install(module, context_manifest, selection_manifest, receipt_dir, **kwargs):
    return Installation(module, context_manifest, selection_manifest, receipt_dir, **kwargs)


def restore(module, handle):
    require(handle.module is module, 'wrong official module for restoration')
    handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', required=True)
    parser.add_argument('--context-manifest', required=True)
    parser.add_argument('--selection-manifest', required=True)
    args = parser.parse_args(argv)
    manifest, blocks, selection, rows = check_context(args.context_manifest, args.selection_manifest)
    print(json.dumps(cb.seal(dict(schema='s2-long-framework-context-check-v1', status='context_files_valid',
        context_sha256=manifest['sha256'], selection_sha256=selection['sha256'],
        task_ids=[row['instance_id'] for row in rows], shared_block_count=len(blocks),
        gpu_initialized=False, model_called=False, actual_sdk_wire_admission_required=True,
        actual_workspace_admission_required=True, new_task_images_and_official_grade_admission_required=True)), indent=2))


if __name__ == '__main__':
    main()
