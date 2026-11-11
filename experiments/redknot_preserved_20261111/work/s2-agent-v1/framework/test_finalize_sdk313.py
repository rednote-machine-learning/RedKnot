import json
from pathlib import Path
import tempfile
import unittest
import ast

from finalize_sdk313 import read_failed
from prewarm_sdk313 import seal, sha


class FinalizeTests(unittest.TestCase):
    def test_both_exact_replay_commands_disable_project_constraints(self):
        for filename in ('finalize_sdk313.py', 'prewarm_sdk313.py'):
            tree=ast.parse((Path(__file__).parent/filename).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='replay-hashed-wheel-bundle']
            self.assertEqual(len(calls),1)
            argv=calls[0].args[1]
            self.assertEqual([n.value for n in argv.elts[1:4]],['--no-config','pip','install'])

    def fixture(self, root):
        names = ('create-env', 'export-frozen', 'locked-build-backends', 'download-locked-artifacts', 'verify-frozen-offline', 'build-locked-wheels', 'build-workspace-wheels', 'official-transformers-pip-fallback')
        steps = []
        for name in names:
            log = root / (name + '.log'); log.write_text('actual-result-fixture\n')
            steps.append({'name': name, 'exit_code': 0, 'timed_out': False, 'log': str(log), 'log_sha256': sha(log)})
        source = {'commit': 'pinned', 'uv_lock_sha256': 'lock', 'tracked_files_sha256': 'files'}
        raw = seal({'schema': 's2-sdk313-prewarm-v1', 'status': 'failed', 'error': 'ValueError: Ambiguous wheel metadata: py.whl', 'source': source, 'source_after': source, 'source_unchanged': True, 'locked_offline_sync_verified': True, 'steps': steps})
        path = root / 'receipt.json'; path.write_text(json.dumps(raw))
        (root / 'progress.json').write_text(json.dumps({'steps': steps}))
        return path, raw

    def test_only_witnessed_failure_with_intact_evidence_is_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            path, raw = self.fixture(Path(d)); before = path.read_bytes()
            self.assertEqual(read_failed(path, raw['sha256']), raw)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                read_failed(path, 'wrong-seal')
            Path(raw['steps'][0]['log']).write_text('changed')
            with self.assertRaises(ValueError):
                read_failed(path, raw['sha256'])

    def test_frozen_sync_failure_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as d:
            path, raw = self.fixture(Path(d)); raw.pop('sha256')
            raw['locked_offline_sync_verified'] = False
            raw = seal(raw); path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                read_failed(path, raw['sha256'])


if __name__ == '__main__':
    unittest.main()
