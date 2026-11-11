"""Run the actual upstream Git/Dockerfile helpers without importing the SDK."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import run_official as adapter


class OfficialCwdTests(unittest.TestCase):
    def helpers(self):
        path=adapter.B/'benchmarks/benchmarks/swebench/build_base_images.py'
        names={'_get_repo_root','_get_sdk_dockerfile','dockerfile_content_hash'}
        source=ast.parse(path.read_text())
        nodes=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in nodes},names)
        namespace=dict(Path=Path,subprocess=subprocess,hashlib=hashlib)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        return namespace

    def test_actual_upstream_helper_resolves_from_non_git_initial_directory(self):
        functions=self.helpers();previous=Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                start=Path(directory).resolve();os.chdir(start)
                with self.assertRaises(subprocess.CalledProcessError):functions['_get_repo_root']()
                with adapter.official_repository_cwd():
                    root=functions['_get_repo_root']().resolve()
                    self.assertEqual(root,(adapter.B/'benchmarks').resolve())
                    dockerfile=functions['_get_sdk_dockerfile']()
                    self.assertTrue(dockerfile.is_file())
                    self.assertEqual(functions['dockerfile_content_hash'](),
                        hashlib.sha256(dockerfile.read_text().encode()).hexdigest()[:7])
                self.assertEqual(Path.cwd(),start)
        finally:os.chdir(previous)

    def test_import_or_main_failure_restores_callers_directory(self):
        previous=Path.cwd()
        with self.assertRaisesRegex(RuntimeError,'official failure'):
            with adapter.official_repository_cwd():
                self.assertEqual(Path.cwd(),(adapter.B/'benchmarks').resolve())
                raise RuntimeError('official failure')
        self.assertEqual(Path.cwd(),previous)


if __name__=='__main__':unittest.main()
