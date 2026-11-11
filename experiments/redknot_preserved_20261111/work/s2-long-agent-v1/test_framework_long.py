"""Focused CPU tests; small fixtures are not long-input benchmark evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import context_builder as cb
import framework_long as fl
import test_context_builder as context_fixtures


class Message:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.fixture = context_fixtures.ContextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.manifest_path, self.selection_path, self.rows, self.manifest = self.fixture.bundle()
        self.root = self.fixture.root
        self.conversations = []
        owner = self

        class Conversation:
            def __init__(self, **kwargs):
                self.id = 'actual-conversation-' + str(len(owner.conversations))
                self.messages, self.runs = [], 0
                owner.conversations.append(self)
            def send_message(self, message, sender=None): self.messages.append((message, sender))
            def run(self): self.runs += 1

        class Evaluation:
            def get_repo_path(self, instance): return '/workspace/django'
            def evaluate_instance(self, instance, workspace):
                conversation = module.Conversation(workspace=workspace)
                conversation.send_message('Official\r\n' + instance.data['problem_statement'])
                conversation.run()
                conversation.send_message('Unmodified official follow-up', sender='official')
                return NS(instance_id=instance.id, marker='original evaluator result')
        module = NS(__file__=__file__, SWEBenchEvaluation=Evaluation, Conversation=Conversation)
        self.module = module
        self.original = module.Conversation, Evaluation.evaluate_instance

    def install(self):
        return fl.install(self.module, self.manifest_path, self.selection_path, self.root/'receipts',
                          expected_source_sha256=cb.file_sha(__file__), message_types=(Message, Message))

    def instance(self, index=0):
        return NS(id=self.rows[index]['instance_id'], data=copy.deepcopy(self.rows[index]))

    def test_two_real_run_paths_keep_issue_first_and_shared_blocks_identical(self):
        handle = self.install()
        with patch.object(fl, 'verify_actual_workspace', return_value={'verified': 'fixture'}) as verify:
            for index in range(2):
                result = self.module.SWEBenchEvaluation().evaluate_instance(self.instance(index), NS())
                self.assertEqual(result.marker, 'original evaluator result')
            self.assertEqual(verify.call_count, 2)
        fl.restore(self.module, handle)
        self.assertEqual((self.module.Conversation, self.module.SWEBenchEvaluation.evaluate_instance), self.original)
        a, b = self.conversations
        self.assertEqual((a.runs, b.runs), (1, 1))
        self.assertNotEqual(a.messages[0][0].content[0].text, b.messages[0][0].content[0].text)
        self.assertEqual([x.text for x in a.messages[0][0].content[1:]],
                         [x.text for x in b.messages[0][0].content[1:]])
        self.assertEqual(a.messages[1], ('Unmodified official follow-up', 'official'))
        self.assertEqual(len(list((self.root/'receipts').rglob('first-message.sent.json'))), 2)

    def test_task_mutation_and_foreign_conversation_fail_closed(self):
        handle = self.install()
        try:
            instance = self.instance()
            instance.data['problem_statement'] += ' changed'
            with self.assertRaisesRegex(ValueError, 'unregistered or modified task'):
                self.module.SWEBenchEvaluation().evaluate_instance(instance, NS())
            with self.assertRaisesRegex(ValueError, 'outside an admitted task'):
                self.module.Conversation()
            self.assertEqual(self.conversations, [])
        finally: handle.close()

    def test_actual_workspace_failure_prevents_send_and_preserves_error(self):
        handle = self.install()
        try:
            with patch.object(fl, 'verify_actual_workspace', side_effect=ValueError('changed source')):
                with self.assertRaisesRegex(ValueError, 'changed source'):
                    self.module.SWEBenchEvaluation().evaluate_instance(self.instance(), NS())
            self.assertEqual(self.conversations[0].messages, [])
            receipt = json.loads(next((self.root/'receipts').rglob('evaluation-returned.json')).read_text())
            self.assertFalse(receipt['first_message_delivered'])
            self.assertIn('changed source', receipt['error'])
            self.assertIsNone(handle.active.get())
        finally: handle.close()

    def test_real_git_blob_and_working_tree_checks(self):
        repo = self.root/'real-repo'
        repo.mkdir()
        subprocess.run(['git','init','-q',str(repo)],check=True)
        source = b'print("natural source")\r\n'
        (repo/'source.py').write_bytes(source)
        subprocess.run(['git','-C',str(repo),'add','source.py'],check=True)
        subprocess.run(['git','-C',str(repo),'-c','user.name=CPU fixture','-c','user.email=fixture@example.invalid',
                        'commit','-qm','fixture'],check=True)
        base = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        blob = hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
        manifest = dict(base_commit=base, files=[dict(path='source.py',sha256=cb.sha(source),git_blob_sha1=blob)])
        class Workspace:
            def execute_command(self, command):
                r = subprocess.run(command,shell=True,capture_output=True,text=True)
                return NS(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
        proof = fl.verify_actual_workspace(Workspace(), str(repo), manifest)
        self.assertEqual(proof['files'][0]['sha256'], cb.sha(source))
        self.assertFalse(proof['task_code_executed'])
        (repo/'source.py').write_text('changed by an earlier session')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            fl.verify_actual_workspace(Workspace(), str(repo), manifest)


if __name__ == '__main__': unittest.main()
