"""Extract agent changes against the admitted official environment baseline.

The official dataset-base-relative patch is preserved verbatim. The returned
patch is a complete git tree diff from the pristine official environment commit
to the same final/index tree. No filename or hunk is filtered; model changes to
setup.py/tox.ini remain changes. Official scoring and test specifications are
not modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shlex
import threading

from offline_grader import (ARCHIVE_SHA, COMMIT, ENVIRONMENT_COMMIT, ENVIRONMENT_TREE,
    INSTANCE, MANIFEST_DIGEST, SETUP_DELTA_SHA, digest, file_sha, read_ref, require, seal)


def load_proof(path, row, *, ref=None):
    path=Path(path).resolve()
    proof=json.loads(path.read_text())
    require(proof.get('sha256') == digest({k:v for k,v in proof.items() if k!='sha256'}),
            'environment admission seal differs')
    if ref is not None:
        require(Path(ref['path']).resolve() == path and ref.get('file_sha256') == file_sha(path)
            and ref.get('sha256',ref.get('seal_sha256')) == proof['sha256'],
            'image provenance is detached from environment admission')
    require((row['instance_id'],row['base_commit']) == (INSTANCE,COMMIT)
        and proof.get('schema') == 's2-official-base-environment-admission-v1'
        and proof.get('status') == 'passed' and proof.get('instance_id') == INSTANCE
        and proof.get('dataset_base_commit') == proof.get('baseline_parent_commit') == COMMIT
        and proof.get('baseline_environment_commit') == ENVIRONMENT_COMMIT
        and proof.get('baseline_environment_tree') == ENVIRONMENT_TREE
        and proof.get('approved_changed_paths') == ['setup.py','tox.ini']
        and proof.get('setup_delta_sha256') == SETUP_DELTA_SHA,
        'only the actual admitted official setup environment is supported')
    require(proof['archive_sha256'].removeprefix('sha256:') == ARCHIVE_SHA
        and proof['registry_manifest_digest'] == MANIFEST_DIGEST
        and file_sha(proof['sif_path']) == proof['sif_sha256'], 'official base image provenance changed')
    read_ref(proof['official_harness_evidence'])
    original=read_ref(proof['original_base_proof_ref'],sealed=True)
    require(original.get('schema') == 's2-official-base-image-cpu-proof-v1'
        and original.get('status') == 'failed' and len(original.get('steps',[])) == 2
        and original['steps'][1]['returncode'] == 0
        and original['steps'][1]['stdout'].strip() == ENVIRONMENT_COMMIT,
        'original strict-HEAD observation was not preserved')
    return seal(dict(schema='s2-patch-baseline-admission-v1',instance_id=INSTANCE,
        dataset_base_commit=COMMIT,environment_commit=ENVIRONMENT_COMMIT,environment_tree=ENVIRONMENT_TREE,
        setup_delta_sha256=SETUP_DELTA_SHA,proof_ref=dict(path=str(path),file_sha256=file_sha(path),sha256=proof['sha256']),
        original_base_proof_ref=proof['original_base_proof_ref'],official_harness_evidence=proof['official_harness_evidence'],
        original_image_sha256=proof['sif_sha256'],model_or_gold_outcomes_used=False))


def _command(workspace, command):
    result=workspace.execute_command(command)
    require(result.exit_code == 0, 'workspace patch-baseline check failed: '+command+'\n'+str(result.stderr))
    return result.stdout


def _identity(evaluator, instance, workspace, admission, *, staged):
    source=shlex.quote(evaluator.get_source_repo_path(instance))
    repo=shlex.quote(evaluator.get_repo_path(instance))
    env=shlex.quote(admission['environment_commit'])
    # The immutable source image and copied repository baseline are checked
    # separately. Agent final HEAD may legitimately include additional commits.
    command=(f'git -C {source} rev-parse HEAD HEAD^{{tree}} && '
        f'git -C {repo} rev-parse {env} {env}^{{tree}} {env}^ && '
        f'git -C {repo} rev-parse HEAD && '+
        (f'git -C {repo} write-tree' if staged else f'git -C {repo} rev-parse HEAD^{{tree}}'))
    lines=_command(workspace,command).splitlines()
    require(len(lines) == 7 and lines[:5] == [admission['environment_commit'],admission['environment_tree'],
        admission['environment_commit'],admission['environment_tree'],admission['dataset_base_commit']],
        'actual image/copied repository environment baseline differs')
    require(all(len(value) == 40 and all(c in '0123456789abcdef' for c in value) for value in lines),
            'actual Git object identity is malformed')
    return dict(source_head=lines[0],source_tree=lines[1],environment_commit=lines[2],environment_tree=lines[3],
        environment_parent=lines[4],head=lines[5],target_tree=lines[6],target='index' if staged else 'HEAD')


class Installation:
    def __init__(self, module, admissions, receipt_dir):
        self.cls=module.SWEBenchEvaluation
        self.originals={name:getattr(self.cls,name) for name in ('get_git_patch','get_staged_git_patch')}
        self.admissions=admissions
        self.directory=Path(receipt_dir).resolve()/'patch-baseline'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.ordinal=0;self.hooks={}
        require(admissions, 'actual environment admissions required')
        for iid,value in admissions.items():
            require(value.get('schema') == 's2-patch-baseline-admission-v1' and value.get('instance_id') == iid
                and value.get('sha256') == digest({k:v for k,v in value.items() if k!='sha256'}),
                'invalid patch baseline admission')
        for name,staged in (('get_git_patch',False),('get_staged_git_patch',True)):
            original=self.originals[name]
            def extract(evaluator,instance,workspace,*,_original=original,_staged=staged):
                return self.extract(evaluator,instance,workspace,_original,_staged)
            self.hooks[name]=extract
            setattr(self.cls,name,extract)

    def extract(self, evaluator, instance, workspace, original, staged):
        iid=instance.data['instance_id']
        require(iid in self.admissions, 'unadmitted extraction task')
        admission=self.admissions[iid]
        require(instance.data['base_commit'] == admission['dataset_base_commit'], 'dataset base changed')
        before=_identity(evaluator,instance,workspace,admission,staged=staged)
        raw=original(evaluator,instance,workspace)
        repo=shlex.quote(evaluator.get_repo_path(instance));env=shlex.quote(admission['environment_commit'])
        command=f'git -C {repo} --no-pager diff --no-color '+(f'--cached {env}' if staged else f'{env} HEAD')
        model=_command(workspace,command)
        after=_identity(evaluator,instance,workspace,admission,staged=staged)
        require(before == after, 'agent tree changed during patch extraction')
        require(type(raw) is str and type(model) is str, 'patch extraction did not return text')
        with self.lock:
            ordinal=self.ordinal;self.ordinal+=1
            stem=f'{ordinal:04d}.'+hashlib.sha256(iid.encode()).hexdigest()[:12]+('.staged' if staged else '.final')
            refs=[]
            for suffix,text in (('dataset-base.patch',raw),('environment-base.patch',model)):
                path=self.directory/(stem+'.'+suffix)
                with path.open('x',encoding='utf-8',newline='') as stream:stream.write(text)
                refs.append(dict(path=str(path),file_sha256=file_sha(path),bytes=path.stat().st_size))
            receipt=seal(dict(schema='s2-environment-relative-patch-v1',instance_id=iid,mode='staged' if staged else 'final',
                dataset_base_commit=admission['dataset_base_commit'],environment_commit=admission['environment_commit'],
                environment_tree=admission['environment_tree'],target_tree=after['target_tree'],actual_git_identity=after,
                baseline_admission_sha256=admission['sha256'],baseline_admission=admission,
                raw_patch_ref=refs[0],model_patch_ref=refs[1],model_patch_command=command,
                raw_official_patch_preserved_verbatim=True,all_paths_retained=True,filename_or_hunk_filtering=False,
                transform='git diff admitted_environment_commit to the same final/index tree',
                extraction_only=True,official_scoring_unchanged=True,gold_or_test_patch_used=False))
            with (self.directory/(stem+'.json')).open('x') as stream:
                json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
        return model


def install(module, admissions, receipt_dir):
    return Installation(module,admissions,receipt_dir)


def restore(module, handle):
    require(module.SWEBenchEvaluation is handle.cls, 'evaluation class ownership changed')
    for name,hook in handle.hooks.items():
        require(getattr(handle.cls,name) is hook, 'patch extraction hook ownership changed')
        setattr(handle.cls,name,handle.originals[name])
