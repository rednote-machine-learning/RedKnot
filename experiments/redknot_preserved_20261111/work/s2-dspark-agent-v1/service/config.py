"""CPU-only sealed configuration for the live S2/Full service."""
from pathlib import Path
import json
import os
import sys
import uuid
from .checkpoint_codec import ENCODER_SHA256
from .native_runtime import digest, seal, require

GATED_SHA = '0321d533deceb984c126b07d24a099316cc7c8dcc769e470ccbcbc7095fa4d36'
NATIVE_SHA = 'd89dad6c23f9df603b7ba27ca3d07717c315bc9d4dcbf7471346552b300cb94f'
GROUP_SHA = '855fbb8e8c02eb7eae9f935472778e53d72b97843e83f3ca0b0736e3074fde35'
IDLE_SHA = 'be34a00fde1262176d6eb69fa50289609343a26b6cf0b5d5fc6d9d4c203e58b3'
RUNTIME_PINS = {
 'strong_native_20261029/native_profile.py':'e68d7d3b39c8255ef1f06d4ba42bb26e2e6121b4f141ace5da06c41060ba0717',
 'strong_native_20261029/strong_native_runner.py':'45f25f878ea62741cef12ca186618483531e0c968a50dc8722051b4177411a7a',
 'kv_recompression_20261029/canonical_request_binding.py':'f25097ab9ebe51662c21b9652af472e43d7664e368276a4c317fc133d02e913e',
 'kv_recompression_20261029/quality_process_group.py':'a7d503f23d1df3f41044c98107b25ca2a09debc1e66384d17d26a9f9f33ba9db',
 'benchmark/pr148_native_runner.py':'615315c76ee54320201daa918d462676cd3c59cb05000769396e3253f0886a47'}


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tree(root, *, python_only=False):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if any(p in ('.git','__pycache__','.pytest_cache') for p in path.relative_to(root).parts):
            continue
        require(not path.is_symlink(), 'source tree symlink: '+str(path))
        if path.is_file() and (not python_only or path.suffix == '.py'):
            result[str(path.relative_to(root))] = file_sha(path)
    return result


def sources(stage, experiment_root):
    stage, root = Path(stage), Path(experiment_root)
    gated = tree(stage/'GatedEngram', python_only=True)
    native = tree(stage/'NativeSnapshot')
    admission = json.loads((stage/'adapted-gated-sources.json').read_text())
    require(admission.get('sha256') == digest({k:v for k,v in admission.items() if k != 'sha256'}), 'adapted Gated source admission seal differs')
    require(admission.get('parent_gated_sha256') == GATED_SHA and admission.get('gated') == gated, 'independent DSpark Gated adapter sources changed')
    require(digest(native) == NATIVE_SHA, 'frozen native snapshot changed/incomplete')
    runtime = {p:file_sha(root/p) for p in RUNTIME_PINS}
    require(runtime == RUNTIME_PINS, 'original native runtime/profile/binding source changed')
    require(file_sha(stage/'idle_optimizer.py') == IDLE_SHA, 'frozen S1.2 idle candidate changed')
    require(file_sha(stage/'profile_group_contract.py') == GROUP_SHA, 'frozen group17 audit helper changed')
    external = {'idle_optimizer.py':IDLE_SHA, 'profile_group_contract.py':GROUP_SHA,
        'adapted-gated-sources.json':file_sha(stage/'adapted-gated-sources.json')}
    for name in ('native_decode_profile.py','dflash_tail_adapter.py','prefix_decode_contract.py'):
        external[name] = file_sha(stage/name)
    for folder in ('service','redknot'):
        external.update({folder+'/'+k:v for k,v in tree(stage/folder, python_only=True).items()
                         if not Path(k).name.startswith('test_')})
    external['service/prefix_descriptor.cpp'] = file_sha(stage/'service/prefix_descriptor.cpp')
    return dict(gated=gated, native=native, runtime=runtime, external=external)


def validate_prefix_bridge(proof, stage):
    require(proof.get('sha256') == digest({k:v for k,v in proof.items() if k != 'sha256'}),
            'prefix host proof seal differs')
    require(proof.get('schema') == 's2-prefix-readonly-host-bridge-proof-v1'
        and proof.get('passed') is True and proof.get('read_only') is True
        and proof.get('actual_native_cpu_structures_checked') is True
        and proof.get('with_cuda') is False and proof.get('cuda_initialized') is False,
        'actual read-only native CPU ABI proof required')
    require(proof['native_source_sha256'] == NATIVE_SHA
        and proof['source_sha256'] == file_sha(Path(stage)/'service/prefix_descriptor.cpp')
        and proof['library_sha256'] == file_sha(proof['library_path']), 'prefix bridge source/binary changed')
    require(proof['sequence_sizeof'] > 0 and proof['metadata_sizeof'] > 0
        and type(proof['torch_cxx11_abi']) is bool and type(proof['torch_version']) is str,
        'prefix bridge ABI identity missing')
    return proof


def weight_inventory(model):
    model = Path(model)
    index = json.loads((model/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    require(shards and all((model/name).is_file() and (model/name).stat().st_size > 0 for name in shards),
            'checkpoint has missing or empty weight shards')
    return {name:dict(bytes=(model/name).stat().st_size, mtime_ns=(model/name).stat().st_mtime_ns) for name in shards}


def build_config(stage, experiment_root, model_path, qualification_manifest, *, admission_mode='representative_session'):
    stage, root, model = map(lambda p:Path(p).resolve(), (stage, experiment_root, model_path))
    manifest = json.loads(Path(qualification_manifest).read_text())
    require(manifest.get('sha256') == digest({k:v for k,v in manifest.items() if k != 'sha256'}), 'qualification source seal differs')
    row = next(r for r in manifest['cohort'] if len(r['input_ids']) >= 16384)
    ids = row['input_ids'][:16384]
    src = sources(stage, root)
    bridge = validate_prefix_bridge(json.loads((stage/'prefix-host-proof.json').read_text()), stage)
    encoder = stage/'checkpoint-encoding'/'encoding.py'
    require(file_sha(encoder) == ENCODER_SHA256, 'V4.1 checkpoint encoder changed')
    model_pins = {name:file_sha(model/name) for name in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json')}
    shards = weight_inventory(model)
    require(model_pins['tokenizer.json'] == manifest['tokenizer']['tokenizer_json_sha256'], 'qualification tokenizer differs')
    execution = digest(dict(sources=src, model=model_pins, weight_inventory=shards, profile='native-nextn5-varlen-main-draft-graph-compact140',
        dp=8,ep=8,cp=1,prefill_chunk=8192,group17=True,memory_fraction=.60,batch_invariant=1,
        max_model_len=270336,max_num_seqs=1,encoder_sha256=ENCODER_SHA256,prefix_host_bridge=bridge))
    return seal(dict(schema='s2-online-service-config-v1', decode_profile='dspark', session_id=uuid.uuid4().hex, stage=str(stage), experiment_root=str(root),
        gated_repo=str(stage/'GatedEngram'), native_repo=str(stage/'NativeSnapshot'), model_path=str(model),
        tokenizer_path=str(model/'tokenizer.json'), encoder_path=str(encoder),
        scope=dict(model_sha256=model_pins['model.safetensors.index.json'],
            tokenizer_sha256=model_pins['tokenizer.json'],execution_sha256=execution),
        source_sha256=src, prefix_host_bridge=bridge, checkpoint_sha256=model_pins, checkpoint_weight_inventory=shards,
        weight_identity_limit='index hash plus shard presence/size/mtime; full weight hashing not performed',
        encoder_sha256=ENCODER_SHA256,
        model_name='deepseek-v4.1-flash', thinking_mode='thinking', reasoning_effort=75,
        qualified_arms=['full','s2','prefix'], dp=8,ep=8,cp=1,concurrency=1,
        qualification_input_ids=ids, qualification_input_ids_sha256=digest(ids),
        qualification_source=dict(manifest_sha256=manifest['sha256'], input_ids_sha256=row['input_ids_sha256'],
            sample_id=row['sample_id'],prefix_tokens=16384,benchmark_data=False),
        startup_timeout_s=1800,request_timeout_s=1800,per_prompt_qualification=admission_mode,
        offline_replay=False,host='127.0.0.1',port=8123,max_service_seconds=7200))


def validate_config(config):
    require(config.get('sha256') == digest({k:v for k,v in config.items() if k != 'sha256'}), 'service config seal differs')
    require(config.get('schema') == 's2-online-service-config-v1' and config.get('qualified_arms') == ['full','s2','prefix']
        and (config.get('dp'),config.get('ep'),config.get('cp'),config.get('concurrency')) == (8,8,1,1), 'service layout differs')
    require(config['host'] == '127.0.0.1' and type(config['port']) is int and 1024 <= config['port'] <= 65535,
            'loopback service address required')
    require(config.get('decode_profile') == 'dspark', 'All inference arms require D-Spark and CUDA Graph')
    require(config['source_sha256'] == sources(config['stage'],config['experiment_root']), 'runtime source map changed')
    validate_prefix_bridge(config['prefix_host_bridge'], config['stage'])
    require(set(config['checkpoint_sha256']) == {'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'}, 'complete checkpoint metadata pins required')
    require(config['checkpoint_weight_inventory'] == weight_inventory(config['model_path']), 'checkpoint weight inventory changed')
    active = {k:v for k,v in os.environ.items() if 'REDKNOT' in k.upper() and k != 'REDKNOT_GLM_RUN_ID' and v.lower() not in ('','0','false','off','none')}
    require(not active, 'unregistered RedKnot environment overrides: '+str(sorted(active)))
    require(config['checkpoint_sha256'] == {n:file_sha(Path(config['model_path'])/n) for n in config['checkpoint_sha256']},
            'checkpoint/tokenizer identity changed')
    require(file_sha(config['encoder_path']) == ENCODER_SHA256 == config['encoder_sha256'], 'encoder identity changed')
    ids = config['qualification_input_ids']
    require(type(ids) is list and len(ids) == 16384 and all(type(t) is int and 0 <= t < 2**31 for t in ids)
        and digest(ids) == config['qualification_input_ids_sha256'], 'native16K qualification tokens changed')
    require(config['per_prompt_qualification'] in ('strict_fresh_dense32_vs_group17','representative_session'), 'unregistered numerical admission')
    require(type(config.get('session_id')) is str and len(config['session_id']) == 32, 'unique service session ID missing')
    require(not any(n == 'torch' or n.startswith('deepinfer') for n in sys.modules), 'CPU admission imported GPU runtime')
    return seal(dict(schema='s2-online-service-cpu-preflight-v1', passed=True, config_sha256=config['sha256'],
        sources_verified=True, gpu_initialized=False, native_source_files=len(config['source_sha256']['native']),
        frozen_native_preserved=True, original_s1_math_with_explicit_dflash_adapter=True, actual_online_service=True, gpu_numerical_qualification_claim=False))
