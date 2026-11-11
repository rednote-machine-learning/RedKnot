import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from sdk_bundle_adapter import adapt_dependency_install, validate_bundle
from prewarm_sdk313 import seal


class BundleTests(unittest.TestCase):
    def test_actual_official_recipe_only_installer_changes(self):
        source = Path(__file__).parent / 'benchmarks/benchmarks/swebench/apptainer_build.py'
        tree = ast.parse(source.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_definition_file_content', '_package_install_script', '_wrap_swebench_deps_script'}]
        ns = {'Path': Path, 'os': os, '_sdk_root': lambda: Path('/sdk')}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
        original = ns['_definition_file_content']('official:image', 'pinned', 'pinned', True, Path('/uv'), None)
        bundle = {'wheel_directory': '/offline/wheels', 'requirements_path': '/offline/requirements.txt', 'receipt_path': '/offline/receipt.json', 'receipt_file_sha256': 'a' * 64, 'post_lock_dependency_changes': {'transformers': {'locked': None, 'final': '4.57.6'}}}
        adapted, proof = adapt_dependency_install(original, bundle)
        self.assertNotIn('uv sync --frozen', adapted)
        self.assertIn('--require-hashes --no-deps', adapted)
        self.assertIn('uv --no-config pip install', adapted)
        self.assertIn('verify_bundle_install.py', adapted)
        self.assertEqual(adapted.split('%environment', 1)[1], original.split('%environment', 1)[1])
        self.assertTrue(proof['sdk_source_and_lock_unchanged'])
        self.assertIn("'docutils<0.21' 'roman'", adapted)
        with self.assertRaises(ValueError):
            adapt_dependency_install(original.replace('--extra boto3', '--extra unknown'), bundle)

    def test_receipt_binds_exact_artifacts_and_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); wheels = root / 'wheels'; wheels.mkdir()
            wheel = wheels / 'x-1-py3-none-any.whl'; wheel.write_bytes(b'fixture')
            h = hashlib.sha256(wheel.read_bytes()).hexdigest()
            req = root / 'requirements.txt'; req.write_text('x==1 --hash=sha256:' + h + '\n')
            body = {'schema': 's2-sdk313-prewarm-v1', 'status': 'complete', 'source_unchanged': True, 'locked_offline_sync_verified': True, 'offline_replay_verified': True, 'source': {'commit': 'pinned', 'uv_lock_sha256': 'lock'}, 'official_transformers_constraint': '>=4.56.0,<5', 'wheel_directory': str(wheels), 'final_requirements': str(req), 'final_requirements_sha256': hashlib.sha256(req.read_bytes()).hexdigest(), 'final_installed': {'packages': {'x': '1'}, 'platform': 'linux', 'machine': 'x86_64', 'python': '3.13.18 fixture'}, 'wheel_inventory': [{'name': 'x', 'version': '1', 'file': wheel.name, 'sha256': h, 'bytes': 7}], 'post_lock_dependency_changes': {}}
            body['dependency_check_passed']=True
            receipt = root / 'receipt.json'; receipt.write_text(json.dumps(seal(body)))
            self.assertEqual(validate_bundle(receipt, 'pinned', 'lock')['wheel_count'], 1)
            with self.assertRaises(ValueError):
                validate_bundle(receipt, 'other', 'lock')
            wheel.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_bundle(receipt, 'pinned', 'lock')


if __name__ == '__main__':
    unittest.main()
