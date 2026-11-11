import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile

import prewarm_sdk313 as p


class PrewarmTests(unittest.TestCase):
    def test_bootstrap_uses_actual_sdk_lock_hashes(self):
        lock = tomllib.loads((p.SDK / 'uv.lock').read_text())
        text = p.lock_bootstrap(lock)
        self.assertIn('setuptools==80.9.0', text)
        self.assertIn('wheel==0.46.3', text)
        self.assertEqual(len(text.splitlines()), 3)
        for line in text.splitlines():
            self.assertIn('--hash=sha256:', line)
        with self.assertRaises(ValueError):
            p.lock_bootstrap({'package': []})

    def test_index_never_accepts_embedded_credentials(self):
        self.assertEqual(p.index_url('https://example.test/simple'), 'https://example.test/simple')
        for value in ('https://user:secret@example.test/simple', 'file:///tmp/index', 'https://example.test/simple?token=secret'):
            with self.assertRaises(Exception):
                p.index_url(value)

    def test_exact_final_wheel_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'foo-1-py3-none-any.whl'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('foo-1.dist-info/METADATA', 'Name: Foo\nVersion: 1\n')
            r = p.wheel_inventory(d, {'foo': '1'})
            self.assertEqual(r[0]['sha256'], p.sha(path))
            with self.assertRaises(ValueError):
                p.wheel_inventory(d, {'foo': '2'})
            duplicate = Path(d) / 'foo-1-py313-none-any.whl'
            duplicate.write_bytes(path.read_bytes())
            with self.assertRaises(ValueError):
                p.wheel_inventory(d, {'foo': '1'})

    def test_nested_vendored_dist_info_is_not_the_wheel_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'py-1.11.0-py2.py3-none-any.whl'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('py-1.11.0.dist-info/METADATA', 'Name: py\nVersion: 1.11.0\n')
                z.writestr('py/_vendored_packages/apipkg-2.0.0.dist-info/METADATA', 'Name: apipkg\nVersion: 2.0.0\n')
                z.writestr('py/_vendored_packages/iniconfig-1.1.1.dist-info/METADATA', 'Name: iniconfig\nVersion: 1.1.1\n')
            self.assertEqual(p.wheel_inventory(d, {'py': '1.11.0'})[0]['name'], 'py')
            with zipfile.ZipFile(path, 'a') as z:
                z.writestr('other-1.dist-info/METADATA', 'Name: other\nVersion: 1\n')
            with self.assertRaises(ValueError):
                p.wheel_inventory(d, {'py': '1.11.0'})

    def test_bounded_real_child_failure_and_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ)
            runner = p.Steps(Path(d), env, 1)
            self.assertFalse(runner.run('timeout', [sys.executable, '-c', 'import time;time.sleep(30)'], required=False))
            record = runner.records[0]
            self.assertTrue(record['timed_out'])
            self.assertEqual(record['timeout_s'], 1)
            self.assertLess(record['elapsed_s'], 7)
            self.assertEqual(json.loads((Path(d) / 'progress.json').read_text())['steps'], runner.records)
            self.assertEqual(env.get('HOME'), os.environ.get('HOME'))
            with self.assertRaises(ProcessLookupError):
                os.kill(record['pid'], 0)


if __name__ == '__main__':
    unittest.main()
