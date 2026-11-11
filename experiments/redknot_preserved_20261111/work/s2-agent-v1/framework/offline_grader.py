"""Run the unchanged official task0 grader from a verified local base SIF.

No model calls. Invoke only after the outer lifecycle has restored the holder.
Only apptainer_eval.image_uri is temporarily adapted; the official TestSpec,
score_instance, score_shell and get_eval_report remain unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
INSTANCE = 'sphinx-doc__sphinx-8638'
COMMIT = '4b452338f914d4f6b54704222d70ae8a746e3db5'
ARCHIVE_SHA = '314b305cf6aef0ad24cc720e2b37943e72977cfdb04b12ac679a406335652ee5'
MANIFEST_DIGEST = 'sha256:3f2d3335613ab578fc026ddf8cda2a835e8f0f24ba6c0605e53241ec518bf78a'
ENVIRONMENT_COMMIT = '27a63c4bbe23e14896382548356d2f9c880812d5'
ENVIRONMENT_TREE = '742277e5b9486c9781a497aad5da88b8b0d5bb96'
SETUP_DELTA_SHA = '5c68e6189821070c3eb4e42feed990474b965d715e0dcda6191f28b98e5e33cd'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_ref(ref, *, sealed=False):
    require(file_sha(ref['path']) == ref['file_sha256'], 'referenced raw evidence bytes changed')
    value = json.loads(Path(ref['path']).read_text())
    if sealed or 'sha256' in ref:
        require(value.get('sha256') == ref['sha256'] == digest({k:v for k,v in value.items() if k != 'sha256'}),
                'referenced raw evidence seal differs')
    return value


def admitted_base(proof_path, registry_manifest, row):
    proof_path, registry_manifest = Path(proof_path).resolve(), Path(registry_manifest).resolve()
    proof = json.loads(proof_path.read_text())
    require(proof.get('sha256') == digest({k:v for k,v in proof.items() if k != 'sha256'}), 'base SIF proof seal differs')
    require((row['instance_id'], row['base_commit']) == (INSTANCE, COMMIT), 'only registered task0 admitted')
    require(proof.get('status') == 'passed' and proof.get('instance_id') == INSTANCE, 'actual base SIF admission missing')
    environment_tree = None
    if proof.get('schema') == 's2-official-base-environment-admission-v1':
        require(proof.get('dataset_base_commit') == proof.get('baseline_parent_commit') == COMMIT
            and proof.get('baseline_environment_commit') == ENVIRONMENT_COMMIT
            and proof.get('baseline_environment_tree') == ENVIRONMENT_TREE
            and proof.get('approved_changed_paths') == ['setup.py','tox.ini']
            and proof.get('setup_delta_sha256') == SETUP_DELTA_SHA, 'official setup-only environment identity differs')
        read_ref(proof['official_harness_evidence'])
        original = read_ref(proof['original_base_proof_ref'], sealed=True)
        require(original.get('schema') == 's2-official-base-image-cpu-proof-v1'
            and original.get('status') == 'failed', 'original strict-HEAD failure must remain raw and preserved')
        steps = original.get('steps', [])
        environment_head, environment_tree = ENVIRONMENT_COMMIT, ENVIRONMENT_TREE
    else:
        require(proof.get('schema') == 's2-official-base-image-cpu-proof-v1'
            and proof.get('base_commit') == COMMIT, 'actual exact-base SIF admission missing')
        steps = proof.get('steps', [])
        environment_head = COMMIT
    require(proof['archive_sha256'].removeprefix('sha256:') == ARCHIVE_SHA and proof['registry_manifest_digest'] == MANIFEST_DIGEST
        and 'sha256:'+file_sha(registry_manifest) == MANIFEST_DIGEST, 'official OCI/archive source identity differs')
    require(len(steps) == 2 and all(step.get('returncode') == 0 for step in steps), 'actual build/commit probe missing')
    image = Path(proof['sif_path']).resolve()
    require(image.is_file() and file_sha(image) == proof['sif_sha256'], 'base SIF bytes changed')
    require(steps[0]['argv'][:2] == ['apptainer','build'] and steps[0]['argv'][2] == str(image)
        and len(steps[0]['argv']) == 4 and steps[0]['argv'][3].startswith('docker-archive:///'),
        'base image was not built directly from local official archive')
    require(steps[1]['argv'][-5:] == ['git','-C','/testbed','rev-parse','HEAD']
        and str(image) in steps[1]['argv'] and steps[1]['stdout'].strip() == environment_head,
        'actual pristine image commit probe differs')
    return dict(image_path=str(image), image_sha256=proof['sif_sha256'], proof=proof,
        expected_environment_commit=environment_head, expected_environment_tree=environment_tree,
        proof_path=str(proof_path), proof_file_sha256=file_sha(proof_path),
        registry_manifest_path=str(registry_manifest), registry_manifest_digest=MANIFEST_DIGEST,
        archive_sha256=ARCHIVE_SHA)


@contextmanager
def local_image_uri(module, row, base):
    """Only replace transport resolution, fail closed for any other task."""
    require(not os.environ.get('OPENHANDS_SWEBENCH_IMAGE_TEMPLATE'), 'custom official image template is not admitted')
    original = module.image_uri
    def local(instance):
        require((instance.get('instance_id'), instance.get('base_commit')) == (row['instance_id'], row['base_commit']),
                'grader requested another task/base commit')
        expected = 'docker://'+row['image_tag_reference'].removeprefix('docker.io/')
        actual = original(instance).replace('docker://docker.io/', 'docker://', 1)
        require(actual == expected, 'official TestSpec resolves a different source image')
        return base['image_path']
    module.image_uri = local
    try:
        yield
    finally:
        require(module.image_uri is local, 'grader image adapter ownership changed')
        module.image_uri = original


def isolate_network():
    """New Linux net namespace; retain only loopback for local test servers."""
    import ctypes
    import fcntl
    import socket
    import struct
    require(sys.platform == 'linux' and os.geteuid() == 0, 'offline grading requires admitted Linux real-root isolation')
    before = os.readlink('/proc/self/ns/net')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x40000000) != 0:  # Linux CLONE_NEWNET
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    after = os.readlink('/proc/self/ns/net')
    require(after != before and socket.if_nameindex() == [(1,'lo')], 'fresh isolated network namespace missing')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack('16sH22x', b'lo', 0)
        flags = struct.unpack('16sH22x', fcntl.ioctl(sock.fileno(), 0x8913, request))[1]
        fcntl.ioctl(sock.fileno(), 0x8914, struct.pack('16sH22x', b'lo', flags | 1))
    return dict(prior_namespace=before, grading_namespace=after,
        interfaces=['lo'], external_network_available=False, loopback_for_local_tests=True)


def verify_fresh_sandbox(module, instance, sandbox, score_dir, cache, *, base=None):
    work = score_dir/instance['instance_id']
    work.mkdir(exist_ok=False)
    head = base['expected_environment_commit'] if base else instance['base_commit']
    tree = base['expected_environment_tree'] if base else None
    tree_check = ('test "$(git rev-parse HEAD^{tree})" = '+shlex.quote(tree)+'; ') if tree else ''
    script = ('set -eu; source /opt/miniconda3/bin/activate; conda activate testbed; cd /testbed; '
        'test "$(git rev-parse HEAD)" = '+shlex.quote(head)+'; '+tree_check+
        'git diff --quiet; git diff --cached --quiet; python --version; command -v python; git rev-parse HEAD')
    command = module.apptainer_base_cmd(sandbox, work)+[script]
    result = module._run(command, work/'base_admission.log', timeout=120, apptainer_cache=cache)
    require(result.returncode == 0, 'fresh grading sandbox base commit/environment differs')
    return dict(dataset_base_commit=instance['base_commit'], baseline_environment_commit=head,
        baseline_environment_tree=tree, tracked_tree_pristine=True,
        conda_testbed_verified=True, stdout=result.stdout, command=command)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--arm', choices=['Full','Prefix','S2'], required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--base-image-proof', type=Path, required=True)
    p.add_argument('--registry-manifest', type=Path, default=B.parent/'container/oci/swebench_sweb.eval.x86_64.sphinx-doc_1776_sphinx-8638/manifest.raw.json')
    p.add_argument('--run', action='store_true')
    args = p.parse_args(argv)
    from run_official import verify_inputs
    framework, selection = verify_inputs()
    row = selection['cases'][0]
    base = admitted_base(args.base_image_proof, args.registry_manifest, row)
    require(args.run_id and Path(args.run_id).name == args.run_id and args.run_id not in ('.','..'), 'run-id must be a single fresh directory name')
    root = args.output_root.resolve()/args.arm/args.run_id
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    require(len(predictions) == 1 and predictions[0].get('instance_id') == INSTANCE
        and set(predictions[0]) <= {'instance_id','model_name_or_path','model_patch'}, 'exactly one official task0 prediction required')
    grade = B.parent/'dataset/grader_only/selected20.jsonl'
    require(file_sha(grade) == selection['files']['grader_only/selected20.jsonl'], 'pinned private grader data changed')
    plan = dict(schema='s2-official-offline-grader-v1', arm=args.arm, run_id=args.run_id,
        instance_id=INSTANCE, base_commit=COMMIT, source_image=base, selection_sha256=selection['sha256'],
        framework_sha256=framework['sha256'], predictions_path=str(args.predictions.resolve()),
        predictions_sha256=file_sha(args.predictions), grader_data_sha256=file_sha(grade), output_root=str(root),
        evaluator='swebench==4.1.0', official_grading_functions_unchanged=True,
        fresh_sandbox_per_arm=True, model_calls=0, gpu_calls=0,
        execution_phase_required='after original holder restoration, ordered by external coordinator',
        status='plan_only')
    if not args.run:
        print(json.dumps(seal(plan), indent=2)); return 0
    require(not root.exists(), 'fresh independent grader output root required')
    plan['network_isolation'] = isolate_network()
    require(importlib.metadata.version('swebench') == '4.1.0', 'official evaluator package pin mismatch')
    sys.path.insert(0, str(B/'benchmarks'))
    from benchmarks.swebench import apptainer_eval as official
    records = [json.loads(line) for line in grade.read_text().splitlines() if line.strip()]
    instance = next(record for record in records if record['instance_id'] == INSTANCE)
    require(instance['base_commit'] == COMMIT, 'grader base commit differs from public task')
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('scores','sandboxes','cache'): (root/folder).mkdir()
    plan['official_apptainer_source_sha256'] = file_sha(official.__file__)
    try:
        with local_image_uri(official, row, base):
            sandbox = official.ensure_sandbox(instance,root/'scores',root/'sandboxes',root/'cache')
            plan['fresh_sandbox_admission'] = verify_fresh_sandbox(official,instance,sandbox,root/'scores',root/'cache',base=base)
            # This unchanged call creates the original TestSpec/eval_script,
            # applies the model patch, runs the official tests and grades them.
            result = official.score_instance(instance=instance,prediction=predictions[0],score_dir=root/'scores',
                sandbox_root=root/'sandboxes',timeout_seconds=3600,apptainer_cache=root/'cache')
        (root/'reports.json').write_text(json.dumps(result,indent=2)+'\n')
        plan.update(status='official_evaluation_returned',reports_sha256=file_sha(root/'reports.json'))
    except BaseException as error:
        plan.update(status='failed',error=repr(error)); raise
    finally:
        (root/'offline-grader-receipt.json').write_text(json.dumps(seal(plan),indent=2)+'\n')
    print(json.dumps(seal(plan),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
