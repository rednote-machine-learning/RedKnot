"""Real local Git fixtures: isolate environment setup without filtering files."""
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest

import patch_baseline as p


def run(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


class Workspace:
    def execute_command(self, command):
        value=subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        return NS(exit_code=value.returncode,stdout=value.stdout,stderr=value.stderr)


class PatchBaselineTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'image';source.mkdir()
        run(source,'init','-q');run(source,'config','user.email','cpu@test.invalid');run(source,'config','user.name','CPU fixture')
        (source/'setup.py').write_text("dependency = 'library'\n")
        (source/'tox.ini').write_text('command = pytest\n')
        (source/'code.py').write_text('answer = 1\n')
        run(source,'add','-A');run(source,'commit','-qm','dataset base');base=run(source,'rev-parse','HEAD')
        (source/'setup.py').write_text("dependency = 'library<2'\n")
        (source/'tox.ini').write_text('command = pytest -rA\n')
        run(source,'commit','-qam','official environment');env=run(source,'rev-parse','HEAD');tree=run(source,'rev-parse','HEAD^{tree}')
        repo=root/'agent';subprocess.run(['git','clone','-q',str(source),str(repo)],check=True)
        run(repo,'config','user.email','cpu@test.invalid');run(repo,'config','user.name','CPU fixture')
        instance=NS(id='fixture-task',data=dict(instance_id='fixture-task',base_commit=base))
        class Evaluation:
            def get_source_repo_path(self,instance):return str(source)
            def get_repo_path(self,instance):return str(repo)
            def get_git_patch(self,instance,workspace):
                return p._command(workspace,f'git -C {shlex.quote(str(repo))} --no-pager diff --no-color {base} HEAD')
            def get_staged_git_patch(self,instance,workspace):
                return p._command(workspace,f'git -C {shlex.quote(str(repo))} --no-pager diff --no-color --cached {base}')
        module=NS(SWEBenchEvaluation=Evaluation)
        admission=p.seal(dict(schema='s2-patch-baseline-admission-v1',instance_id='fixture-task',
            dataset_base_commit=base,environment_commit=env,environment_tree=tree,cpu_fixture=True))
        handle=p.install(module,{'fixture-task':admission},root/'receipts')
        self.addCleanup(p.restore,module,handle)
        return repo,source,instance,Evaluation(),Workspace(),root/'receipts/patch-baseline'

    def audit(self,directory):
        return json.loads(next(directory.glob('*.json')).read_text())

    def test_untouched_agent_returns_empty_and_preserves_original_setup_patch(self):
        with tempfile.TemporaryDirectory() as name:
            repo,source,instance,evaluator,workspace,out=self.fixture(Path(name))
            result=evaluator.get_git_patch(instance,workspace)
            self.assertEqual(result,'')
            report=self.audit(out);raw=Path(report['raw_patch_ref']['path']).read_text()
            self.assertIn('setup.py',raw);self.assertIn('tox.ini',raw)
            self.assertTrue(report['raw_official_patch_preserved_verbatim'])
            self.assertEqual(report['target_tree'],run(repo,'rev-parse','HEAD^{tree}'))
            self.assertEqual(Path(report['model_patch_ref']['path']).read_text(),result)

    def test_source_fix_is_retained_and_applies_to_the_same_official_environment(self):
        with tempfile.TemporaryDirectory() as name:
            repo,source,instance,evaluator,workspace,out=self.fixture(Path(name))
            (repo/'code.py').write_text('answer = 2\n');run(repo,'commit','-qam','agent fix')
            result=evaluator.get_git_patch(instance,workspace)
            self.assertIn('code.py',result);self.assertNotIn('setup.py',result)
            raw=Path(self.audit(out)['raw_patch_ref']['path']).read_text();self.assertIn('setup.py',raw)
            applied=Path(name)/'grade';subprocess.run(['git','clone','-q',str(source),str(applied)],check=True)
            subprocess.run(['git','-C',str(applied),'apply','--check','-'],input=result,text=True,check=True)
            subprocess.run(['git','-C',str(applied),'apply','-'],input=result,text=True,check=True)
            run(applied,'add','-A')
            self.assertEqual(run(applied,'write-tree'),run(repo,'rev-parse','HEAD^{tree}'))

    def test_legitimate_agent_setup_and_tox_changes_are_never_dropped(self):
        with tempfile.TemporaryDirectory() as name:
            repo,source,instance,evaluator,workspace,out=self.fixture(Path(name))
            (repo/'setup.py').write_text("dependency = 'library<3'\n")
            (repo/'tox.ini').write_text('command = pytest -rA --strict\n')
            run(repo,'commit','-qam','agent edits setup')
            result=evaluator.get_git_patch(instance,workspace)
            self.assertIn('-dependency = \'library<2\'',result);self.assertIn('+dependency = \'library<3\'',result)
            self.assertIn('tox.ini',result);self.assertIn('--strict',result)
            self.assertFalse(self.audit(out)['filename_or_hunk_filtering'])

    def test_failure_staged_tree_includes_new_files_and_model_setup_changes(self):
        with tempfile.TemporaryDirectory() as name:
            repo,source,instance,evaluator,workspace,out=self.fixture(Path(name))
            (repo/'new.py').write_text('new_function = 3\n')
            (repo/'setup.py').write_text("dependency = 'library<4'\n")
            run(repo,'add','-A')
            result=evaluator.get_staged_git_patch(instance,workspace)
            self.assertIn('new.py',result);self.assertIn('library<4',result)
            report=self.audit(out);self.assertEqual(report['mode'],'staged')
            self.assertEqual(report['target_tree'],run(repo,'write-tree'))

    def test_wrong_actual_image_head_is_refused_without_returning_patch(self):
        with tempfile.TemporaryDirectory() as name:
            repo,source,instance,evaluator,workspace,out=self.fixture(Path(name))
            run(source,'commit','--allow-empty','-qm','unexpected image modification')
            with self.assertRaisesRegex(ValueError,'baseline differs'):evaluator.get_git_patch(instance,workspace)
            self.assertEqual(list(out.glob('*.json')),[])


if __name__=='__main__':unittest.main()
