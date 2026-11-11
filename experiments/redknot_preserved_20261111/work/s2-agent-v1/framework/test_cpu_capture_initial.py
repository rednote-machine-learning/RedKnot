import hashlib
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import os
import ast
import json
import sys

import cpu_capture_initial
from cpu_capture_initial import verify_mock_patch


class Workspace:
    def execute_command(self, command):
        p=subprocess.run(shlex.split(command),capture_output=True,text=True)
        return SimpleNamespace(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)


class CapturePatchTests(unittest.TestCase):
    def test_actual_official_instruction_accepts_capture_details_and_matches_sealed_prompt(self):
        # Execute the pinned function, not a reimplementation of its assertion
        # or template. The local test runtime keeps Jinja in its private deps.
        try:
            from jinja2 import Environment,FileSystemLoader
        except ImportError:
            sys.path.insert(0,str(Path(__file__).parent/'_deps'))
            from jinja2 import Environment,FileSystemLoader
        b=Path(__file__).parent
        upstream=b/'benchmarks/benchmarks/swebench/run_infer.py'
        tree=ast.parse(upstream.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        official_metadata=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='EvalMetadata')
        capture_tree=ast.parse((b/'cpu_capture_initial.py').read_text())
        actual_metadata=next(n for n in ast.walk(capture_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='EvalMetadata')
        actual_details=ast.literal_eval(next(k.value for k in actual_metadata.keywords if k.arg=='details'))
        official_details=ast.literal_eval(next(k.value for k in official_metadata.keywords if k.arg=='details'))
        self.assertEqual(actual_details,official_details)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_instruction')
        ns={'os':os,'EvalMetadata':SimpleNamespace,'Environment':Environment,'FileSystemLoader':FileSystemLoader}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(upstream),'exec'),ns)
        row=json.loads((b.parent/'dataset/agent_visible/tasks20.jsonl').read_text().splitlines()[0]);row['repo_path']='/workspace/sphinx/'
        metadata=SimpleNamespace(details=actual_details,prompt_path=str(b/'benchmarks/benchmarks/swebench/prompts/default.j2'))
        rendered=ns['get_instruction'](row,metadata,'/workspace')
        sealed=(b.parent/'dataset/agent_visible/prompts/sphinx-doc__sphinx-8638.txt').read_bytes()
        self.assertEqual(rendered.encode('utf-8'),sealed)
        metadata.details=None
        with self.assertRaises(AssertionError):ns['get_instruction'](row,metadata,'/workspace')

    def test_entire_capture_uses_official_repository_cwd_and_restores(self):
        previous=Path.cwd()
        with patch.object(cpu_capture_initial,'_capture',side_effect=lambda *args: str(Path.cwd())):
            actual=cpu_capture_initial.capture(None,0,'unused')
        self.assertEqual(Path(actual),Path(__file__).parent/'benchmarks')
        self.assertEqual(Path.cwd(),previous)
        with patch.object(cpu_capture_initial,'_capture',side_effect=ValueError('fixture')):
            with self.assertRaises(ValueError):cpu_capture_initial.capture(None,0,'unused')
        self.assertEqual(Path.cwd(),previous)

    def fixture(self, root):
        repo=root/'repo';repo.mkdir()
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()
        git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','CPU fixture')
        (repo/'setup.py').write_text('base dependencies\n');git('add','setup.py');git('commit','-m','dataset base');base=git('rev-parse','HEAD')
        (repo/'setup.py').write_text('official environment dependencies\n');git('commit','-am','official environment');env=git('rev-parse','HEAD')
        raw=subprocess.check_output(['git','-C',str(repo),'--no-pager','diff','--no-color',base,'HEAD'],text=True)
        admission={'environment_commit':env,'setup_delta_sha256':hashlib.sha256(raw.encode()).hexdigest()}
        return repo,git,{'base_commit':base},admission,raw

    def test_noop_preserves_exact_official_delta_and_empty_model_patch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo,git,row,admission,raw=self.fixture(root)
            r=verify_mock_patch(Workspace(),str(repo),row,admission,SimpleNamespace(test_result={'git_patch':''}),root)
            self.assertTrue(r['canonical_patch_empty']);self.assertFalse(r['scored'])
            self.assertEqual(Path(r['raw_patch_path']).read_text(),raw)
            with self.assertRaises(ValueError):
                verify_mock_patch(Workspace(),str(repo),row,None,SimpleNamespace(test_result={'git_patch':''}),root)

    def test_model_changes_to_setup_file_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo,git,row,admission,raw=self.fixture(root)
            (repo/'setup.py').write_text('model changed setup.py\n');git('commit','-am','model change')
            with self.assertRaises(ValueError):
                verify_mock_patch(Workspace(),str(repo),row,admission,SimpleNamespace(test_result={'git_patch':''}),root)


if __name__=='__main__':unittest.main()
