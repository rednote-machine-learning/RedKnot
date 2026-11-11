"""CPU selection and actual publication-race regressions; no GPU execution."""
import ast
from copy import deepcopy
from functools import lru_cache
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import run_s12_reproduction as runner
HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def fixture_manifest():
    """Use the exact original full manifest fetched for the retained package."""
    return runner.read_sealed(HERE/'inputs-128-original/manifest.json')


class SelectionTests(unittest.TestCase):
    def test_import_has_no_cuda_or_execution_side_effect(self):
        code='import sys; import run_s12_reproduction; assert not any(k=="torch" or k.startswith("torch.") or k=="deepinfer" or k.startswith("deepinfer.") for k in sys.modules)'
        subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,check=True,capture_output=True)

    def test_all_lengths_and_counts_use_original_cohort_order(self):
        manifest=fixture_manifest()
        for band in ('64k','128k','both'):
            for count in (1,3,10):
                rows=runner.selected_rows(manifest,length=band,cases_per_length=count)
                bands=('64k','128k') if band=='both' else (band,)
                expected=[index for length in bands for index in runner.BAND_INDICES[length][:count]]
                self.assertEqual([row['index'] for row in rows],expected)
                self.assertEqual([runner.case_pin(row)['length_band'] for row in rows],
                    [length for length in bands for _ in range(count)])

    def test_foreign_cohort_and_token_changes_are_rejected(self):
        manifest=fixture_manifest()
        for length,count in [('32k',1),('128k',0),('both',11)]:
            with self.assertRaises(ValueError):runner.selected_rows(manifest,length=length,cases_per_length=count)
        position=next(i for i,row in enumerate(manifest['cohort']) if row['index']==3)
        row=deepcopy(manifest['cohort'][position]);row['input_ids'][0]+=1
        rows=list(manifest['cohort']);rows[position]=row;broken=dict(manifest,cohort=rows)
        with self.assertRaisesRegex(ValueError,'token input changed'):
            runner.selected_rows(broken,length='64k',cases_per_length=1)

    def test_config_has_no_historical_admission_or_expired_deadline(self):
        manifest=fixture_manifest();protocol=seal({'schema':'input-fixture'})
        with patch.object(runner,'read_sealed',side_effect=[manifest,protocol]):
            config=runner.build_config('manifest','protocol',length='both',cases_per_length=1)
        self.assertEqual(config['evaluation_case_indices'],[3,2])
        self.assertEqual(config['lifecycle_timeout_seconds'],7200)
        self.assertEqual(config['timeout_origin'],'actual_gpu_release_of_this_explicit_reproduction')
        self.assertTrue(config['case_publication_barrier_before_all_rank_reference'])
        self.assertFalse(config['historical_64k_gpu_measurements_used'])
        self.assertEqual(set(config['external_source_sha256']),{'run_s12_reproduction.py','reproduction_contract.py',
            'idle_optimizer.py','profile_group_contract.py'})
        self.assertFalse(any('predecessor' in key or 'deadline' in key for key in config))

    def test_rotation_is_fresh_for_each_explicit_selection(self):
        indices=[3,5,2,4]
        for ordinal,index in enumerate(indices):
            schedule=runner.timing_schedule(index,indices)
            self.assertEqual(schedule[0],(0,runner.ARMS[ordinal%2]))
            self.assertEqual(set(schedule),{(repeat,arm) for repeat in range(3) for arm in runner.ARMS})
        with self.assertRaises(ValueError):runner.timing_schedule(12,indices)

    def test_frozen_operators_match_accepted_s12(self):
        from gpu_pilot.launch_pilot import source_files
        self.assertEqual(runner.digest(source_files(runner.STAGE)),runner.FROZEN_STAGE_SOURCES_SHA256)
        self.assertEqual({name:runner.file_digest(HERE/name) for name in runner.UPDATED_SYSTEM_SOURCES},runner.FROZEN_SYSTEM_SHA256)
        self.assertEqual({name:runner.file_digest(HERE/name) for name in runner.FROZEN_EXTERNAL_SHA256},runner.FROZEN_EXTERNAL_SHA256)

def seal(body):
    value={k:v for k,v in body.items() if k!='sha256'}
    return dict(value,sha256=runner.digest(value))

def actual_publication_block(*,remove_guard=False):
    """Execute the actual worker's publication statements in CPU rank threads.

    Extracting this small source block avoids loading the GPU worker. The
    regression uses real files and the original evidence_ref implementation;
    it is not a second hand-written copy of the intended publication order.
    """
    tree=ast.parse(inspect.getsource(runner.worker_impl))
    loops=[node for node in ast.walk(tree) if isinstance(node,ast.For)
        and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name)
        and node.iter.func.id=='selected_rows']
    assert len(loops)==1
    body=loops[0].body
    starts=[i for i,node in enumerate(body) if isinstance(node,ast.If)
        and 'write_json' in ast.unparse(node) and '.clean-paired.json' in ast.unparse(node)]
    assert len(starts)==1
    block=deepcopy(body[starts[0]:])
    assert len(block)==4 and ast.unparse(block[1])=='driver.barrier()'
    assert ast.unparse(block[3])=='driver.barrier()'
    if remove_guard:del block[1]
    function=ast.parse('def publication(rank, driver, args, prefix, clean, index, completed):\n    pass\n').body[0]
    function.body=[*block,ast.Return(value=ast.Name(id='completed',ctx=ast.Load()))]
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace={'write_json':runner.write_json,'evidence_ref':runner.evidence_ref}
    exec(compile(module,'<actual-worker-publication-block>','exec'),namespace)
    return namespace['publication']


class PublicationTests(unittest.TestCase):
    def test_actual_eight_rank_publication_waits_for_late_rank0_writer(self):
        publication=actual_publication_block();barrier=threading.Barrier(8)
        waiting=set();lock=threading.Lock();nonzero_waiting=threading.Event()
        results={};errors={}
        class Driver:
            def __init__(self,rank):self.rank=rank;self.calls=0
            def barrier(self):
                self.calls+=1
                if self.calls==1 and self.rank:
                    with lock:
                        waiting.add(self.rank)
                        if len(waiting)==7:nonzero_waiting.set()
                barrier.wait(timeout=5)
        with tempfile.TemporaryDirectory() as directory:
            args=NS(out_dir=Path(directory));clean=seal(dict(index=14,scope='publication-regression'))
            def rank_task(rank):
                try:results[rank]=publication(rank,Driver(rank),args,'case14',clean,14,[])
                except BaseException as exc:errors[rank]=exc
            # Deliberately let all seven readers arrive before the writer.
            threads=[threading.Thread(target=rank_task,args=(rank,),daemon=True) for rank in range(1,8)]
            for thread in threads:thread.start()
            try:
                self.assertTrue(nonzero_waiting.wait(timeout=3),'readers did not wait at the publication barrier')
                self.assertFalse((args.out_dir/'case14.clean-paired.json').exists())
                self.assertEqual(errors,{})
                writer=threading.Thread(target=rank_task,args=(0,),daemon=True);threads.append(writer);writer.start()
                for thread in threads:thread.join(timeout=5)
                self.assertTrue(all(not thread.is_alive() for thread in threads))
                self.assertEqual(errors,{})
                self.assertEqual(set(results),set(range(8)))
                self.assertTrue(all(value==results[0] for value in results.values()))
                self.assertEqual(results[0][0]['clean_case']['seal_sha256'],clean['sha256'])
            finally:
                barrier.abort()
                for thread in threads:thread.join(timeout=1)

    def test_removing_new_barrier_reproduces_all_seven_actual_failures(self):
        publication=actual_publication_block(remove_guard=True)
        errors={}
        with tempfile.TemporaryDirectory() as directory:
            args=NS(out_dir=Path(directory));clean=seal(dict(index=14,scope='publication-regression'))
            def rank_task(rank):
                try:
                    publication(rank,NS(barrier=Mock(side_effect=AssertionError('reader reached later barrier'))),
                        args,'case14',clean,14,[])
                except BaseException as exc:errors[rank]=exc
            threads=[threading.Thread(target=rank_task,args=(rank,)) for rank in range(1,8)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(timeout=3)
            self.assertEqual(set(errors),set(range(1,8)))
            self.assertTrue(all(type(exc) is ValueError and str(exc)=='raw evidence must exist before closure'
                for exc in errors.values()))
