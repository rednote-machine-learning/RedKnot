"""Explicit CPU C++ build/probe for the read-only native cache descriptor."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

from .config import file_sha, tree, NATIVE_SHA
from .native_runtime import digest, seal, write_json, require


def import_bridge(proof):
    require(file_sha(proof['library_path']) == proof['library_sha256'], 'prefix host library changed')
    spec = importlib.util.spec_from_file_location(proof['module_name'],proof['library_path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(native_repo, build_dir):
    native_repo, build_dir = Path(native_repo).resolve(), Path(build_dir).resolve()
    require(digest(tree(native_repo)) == NATIVE_SHA, 'native headers/source changed')
    build_dir.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(native_repo))
    import torch
    require(not torch.cuda.is_initialized(),'CPU bridge build found initialized CUDA')
    import numpy as np
    from deepinfer._C.engine import Sequence, SequenceData, SequenceMetadata
    from deepinfer.sampling_params import SamplingParams
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('prefix_descriptor.cpp')
    name = 's2_prefix_descriptor_'+file_sha(source)[:12]
    module = load(name=name, sources=[str(source)], extra_include_paths=[str(native_repo/'csrc')],
        extra_cflags=['-O2','-std=c++20'], build_directory=str(build_dir), with_cuda=False, verbose=True)
    params = SamplingParams(temperature=0.0,max_tokens=32,ignore_eos=False)
    ids = np.asarray([0,11,22,33],dtype=np.int32)
    seq = Sequence(seq_id=31,prompt_token_ids=ids,compressed_prompt_token_ids=ids,
        multi_modal_data=None,block_size=128,request_id='cpu-prefix-abi',user_id='test',
        sampling_params=params,arrival_time=0.,golden_token_ids=[])
    expected = dict(request_id='cpu-prefix-abi',block_table=[],num_cached_tokens=0,state_id=-1,
        swa_valid_start=0,block_size=128,last_copy_state_token_idx=0,prompt_tokens=4,
        previous_num_tokens=0,output_tokens=0,schedule_status='WAITING',total_inflight_tokens=0,
        last_inflight_tokens=0,mtp_info_slot=0,mtp_has_draft_info=False,mtp_last_output_token=33,
        mtp_accepted_count=0,mtp_draft_count=0,mtp_accept_rate_count=0,mtp_prev_accept_rate_count=0,
        mtp_confidence_count=0,mtp_prev_confidence_count=0,mtp_process_count=0,mtp_schedule_count=0)
    require(module.sequence_descriptor(seq) == expected,'native Sequence ABI descriptor mismatch')
    seq.state_id, seq.num_cached_tokens, seq.swa_valid_start = 19,2,1
    changed = module.sequence_descriptor(seq)
    require(changed == dict(expected,state_id=19,num_cached_tokens=2,previous_num_tokens=2,swa_valid_start=1),
            'read-only bridge does not observe actual registered native setters')
    data = SequenceData([0,11,22,33])
    meta = SequenceMetadata('cpu-prefix-abi',31,True,data,data,None,params,[3,7],128,256,
        -1,[],[],0,19)
    require(module.metadata_descriptor(meta) == dict(request_id='cpu-prefix-abi',block_table=[3,7],
        state_id=19,swa_valid_start=0,is_prompt=True,is_placeholder=False,seqlen_q=128,seqlen_k=256,
        mtp_info_read_slot=0,mtp_info_write_slot=0),
        'native SequenceMetadata ABI descriptor mismatch')
    require(not torch.cuda.is_initialized(),'CPU bridge build initialized CUDA')
    path = Path(module.__file__).resolve()
    return seal(dict(schema='s2-prefix-readonly-host-bridge-proof-v1',passed=True,
        source_path=str(source),source_sha256=file_sha(source),library_path=str(path),library_sha256=file_sha(path),
        module_name=name,native_source_sha256=NATIVE_SHA,torch_version=torch.__version__,
        torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,with_cuda=False,cuda_initialized=False,
        sequence_sizeof=module.sequence_sizeof,metadata_sizeof=module.metadata_sizeof,
        descriptor_contract=module.descriptor_contract,
        fresh_mtp_state_checked=True,cache_setter_rollback_baseline_checked=True,
        actual_native_cpu_structures_checked=True,read_only=True))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-repo',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=build(a.native_repo,a.build_dir)
    write_json(a.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
