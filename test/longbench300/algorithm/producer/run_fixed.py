"""Full / S1.6 with runtime-M SM90 GEMM on the matched native LongBench B4 workload."""
import argparse
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REVISION = 's16-sm90-runtime-m-asm-ask-v1'
KERNEL_FILES = ('install_dynamic_gemm.py', 'dynamic_kernel.py')
HERE = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base(path):
    require(Path(path).is_file(), f'Missing parent RedKnot runner: {path}')
    spec = importlib.util.spec_from_file_location('redknot_frozen_s16_main', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def paths(data, out, runner=None, kernel=None):
    data, out = Path(data).resolve(), Path(out).resolve()
    old = data.parent/'results'
    require(out != old and old not in out.parents and out not in old.parents
            and out != data and data not in out.parents and out not in data.parents,
            'Use new results; old results and the input data must stay isolated')
    parent = data.parent/'sources/runtime/run_s16.py' if runner is None else Path(runner)
    k = HERE.parent.parent/'kernel' if kernel is None else Path(kernel)
    return data, out, parent.resolve(), k.resolve()


def identity(base, runner, kernel, data):
    runner_sha = sha(runner)
    return dict(gemm_revision=REVISION, parent_runner_sha256=runner_sha,
                runner_sha256=sha(HERE), kernel_sha256={n:sha(kernel/n) for n in KERNEL_FILES},
                parent_template_sha256=base.ARTIFACT, manifest=str(data),
                parent_contract_id=base.digest(dict(template=base.ARTIFACT,runner_sha256=runner_sha,
                    data_root=str(data),methods=base.ARMS,owner_count=4)),
                methods=base.ARMS, owner_count=4,
                dynamic_scope='complete online S1.6 request; Full and offline use original GEMM')


def scope_phase(original, handle, gather, rank, revision):
    def run(native, owners, phase, method, mode, limits, documents):
        if mode == 'capture_independent':
            require(method == 'Full' and handle.active is handle.native, 'Capture must use original GEMM')
            return original(native, owners, phase, method, mode, limits, documents)
        enabled = method == 'Cone15' and mode == 'sensitive'
        require(phase.endswith('.quality') and (enabled or (method == 'Full' and mode is None)),
                'Unexpected S1.6 production method/phase')
        require(handle.active is handle.native, 'Previous online phase did not restore native GEMM')
        handle.set_dynamic(enabled)
        try:
            before = handle.stats()
            result = original(native, owners, phase, method, mode, limits, documents)
            after = handle.stats()
            states = gather(dict(rank=rank, original_variants_before=before['original_variants'],
                                 original_variants_after=after['original_variants'],
                                 dynamic_variants_before=before['dynamic_variants'],
                                 dynamic_variants_after=after['dynamic_variants'], enabled=after['enabled']))
        finally:
            handle.set_dynamic(False)
        require([x['rank'] for x in states] == list(range(8)), 'Incomplete rank GEMM measurements')
        for x in states:
            require(x['enabled'] is enabled and x['dynamic_variants_before'] <= x['dynamic_variants_after']
                    and (enabled or x['dynamic_variants_before'] == x['dynamic_variants_after']),
                    'Unexpected native/dynamic GEMM variant growth')
        if enabled:
            require(any(states[r]['dynamic_variants_after'] > 0 for r in owners),
                    'Candidate GEMM never observed in any actual owner JIT')
        for req in result:
            state = dict(states[req['owner_rank']])
            state['new_dynamic_variants'] = state['dynamic_variants_after'] - state['dynamic_variants_before']
            req['gemm'] = dict(state, gemm_revision=revision, method_uses_dynamic_gemm=enabled,
                               variant_count_scope='process-local cache cardinality; a cache hit can add zero variants')
        return result
    return run


def worker(options, contract, attempt, world, rank):
    base = load_base(options['runner'])
    kernel, data, out = (Path(options[k]) for k in ('kernel','data','out'))
    stamp = identity(base, Path(options['runner']), kernel, data)
    require(base.digest(stamp) == contract, 'Producer sources changed before worker creation')
    previous = base.process
    original_save = base.save
    old_phase = base.execute_phase

    def write(path, obj):
        if obj.get('method') in base.ARMS and obj.get('schema') == 's16-longbench-b4-result-v1':
            obj = dict(obj, gemm_revision=REVISION, parent_runner_sha256=stamp['parent_runner_sha256'],
                       parent_contract_id=stamp['parent_contract_id'], kernel_sha256=stamp['kernel_sha256'],
                       gemm_scope=stamp['dynamic_scope'])
        if Path(path).name == 'group.json':
            obj = dict(obj, gemm_revision=REVISION, parent_runner_sha256=stamp['parent_runner_sha256'])
        return original_save(path, obj)

    def process(native, manifest, output, active_contract, *args, **kwargs):
        sys.path.insert(0, str(kernel))
        from install_dynamic_gemm import install_dynamic_gemm
        require(Path(sys.modules['install_dynamic_gemm'].__file__).resolve() == kernel/'install_dynamic_gemm.py',
                'Different process-local kernel installer')
        handle = install_dynamic_gemm(enabled=False)
        base.execute_phase = scope_phase(old_phase,handle,native.driver.gather,native.rank,REVISION)
        try:
            return previous(native,manifest,output,active_contract,*args,**kwargs)
        finally:
            base.execute_phase = old_phase
            handle.restore()

    base.save = write
    base.process = process
    try:
        base.worker(options['template'],str(data),str(out),str(attempt),contract,world,rank)
    finally:
        base.save = original_save
        base.process = previous


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('template','manifest','output-dir'):
        cli.add_argument('--'+name,type=Path,required=True)
    for name in ('ready-file','release-file','runner','kernel'):
        cli.add_argument('--'+name,type=Path)
    action=cli.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-only',action='store_true')
    action.add_argument('--run',action='store_true')
    args=cli.parse_args()
    data,out,runner,kernel=paths(args.manifest,args.output_dir,args.runner,args.kernel)
    base=load_base(runner)
    template=args.template.resolve()
    cfg=base.load_template(template)
    stamp=identity(base,runner,kernel,data)
    contract=base.digest(stamp)
    require(data.is_dir(),'Input manifest directory missing')
    if cfg.get('strict_source_hashes', False):
        require(sha(cfg['adapter_ref']['path'])==cfg['adapter_ref']['file_sha256'] and
                sha(cfg['service_config_ref']['path'])==cfg['service_config_ref']['file_sha256'],'Changed native adapter/service')
        for source,expected in cfg['source_sha256'].items():
            require(sha(source)==expected,'Changed native source '+source)
    sys.path.insert(0,cfg['fixed_pack_runtime_root'])
    import fixed_pack_runtime_adapter as adapter
    preserved,_=adapter.load_dependencies(cfg)
    adapter.private_preserved_module(preserved)
    require('torch' not in sys.modules and 'deepinfer' not in sys.modules,'GPU libraries loaded before allocation')
    if args.check_only:
        first=next(iter(sorted((data/'groups').glob('*.json'))),None)
        if first:
            base.group_rows(data,first)
        print(json.dumps(dict(status='cpu_pass',contract_id=contract,**stamp,model_calls=0)),flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        first=out/'gemm-revision.json'
        if first.exists():
            meta=base.read(first)
            require(meta.pop('sha256')==base.digest(meta) and meta==dict(contract_id=contract,**stamp),
                    'Existing output has a different frozen producer')
        else:
            require(not any(p.name!='runner.lock' for p in out.iterdir()),'Output already belongs to an earlier run')
            base.save(first,dict(contract_id=contract,**stamp))
        require(not (out/'run.json').exists(),'New experiment is already complete')
        sys.path.insert(0,cfg['rag_runtime_root'])
        from run_rag_dspark_pair import wait_release
        wait_release(args.ready_file,args.release_file,contract)
        root=Path(base.read(cfg['service_config_ref']['path'])['experiment_root'])/'kv_recompression_20261029'
        launcher=root/'quality_process_group.py'
        require(sha(launcher)==cfg['source_sha256'][str(launcher)],'Changed native launcher')
        sys.path.insert(0,str(root))
        from quality_process_group import launch
        attempt=out/'attempts'/f'run-{time.time_ns()}'
        attempt.mkdir(parents=True)
        options=dict(template=str(template),kernel=str(kernel),runner=str(runner),data=str(data),out=str(out))
        launch(functools.partial(worker,options,contract,attempt),8,attempt)


if __name__=='__main__':
    main()
