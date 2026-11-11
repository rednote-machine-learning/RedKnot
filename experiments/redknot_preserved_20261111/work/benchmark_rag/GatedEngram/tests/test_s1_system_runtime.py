"""CPU tests of real controller/worker scheduling, never a GPU parity claim."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from kv_cache_engram.core import token_digest
from kv_cache_engram.s1_runtime import S1Controller
from kv_cache_engram.s1_system_batch import batch_schedule
from kv_cache_engram.s1_system_runtime import (S1SystemController, install_worker_batch_schedule,
                                             classify_world_step, inactive_empty_query_cu)
from gpu_pilot.index_hit20_empty_ep_qualification import EmptyEPCapability
from tests.test_s1_system_attention import np, torch

if np is not None:
    from tests.test_s1_system_attention import Array, NumpyTorch
    class RuntimeArray(Array):
        def is_contiguous(self): return self.flags.c_contiguous
        def clone(self): return self.copy().view(RuntimeArray)
        def transpose(self, dim0, dim1): return np.swapaxes(self,dim0,dim1).view(RuntimeArray)
        def new_empty(self, shape, dtype=None):
            return np.empty(shape,dtype=dtype or self.dtype).view(RuntimeArray)
        def new_tensor(self, values):
            return np.array(values,dtype=self.dtype).view(RuntimeArray)
        def new_zeros(self, shape): return np.zeros(shape,dtype=self.dtype).view(RuntimeArray)
    class RuntimeNumpy(NumpyTorch):
        bfloat16, float8_e4m3fn = np.float32, np.float32
        @staticmethod
        def tensor(values, *, device=None, dtype=None): return np.array(values,dtype=dtype).view(RuntimeArray)
        @staticmethod
        def empty(shape, *, device=None, dtype=None): return np.empty(shape,dtype=dtype).view(RuntimeArray)
        @staticmethod
        def zeros(shape, *, device=None, dtype=None): return np.zeros(shape,dtype=dtype).view(RuntimeArray)
        @staticmethod
        def empty_like(value, *, dtype=None): return np.empty_like(value,dtype=dtype).view(RuntimeArray)


CrossLayerRuntime = NS
Movable = lambda value: value
tile_kernels = NS(engram=NS(engram_hash=lambda history,*args: history))


class FakeMLP:
    hidden_size = 5120
    def __init__(self, layer, calls): self.layer, self.calls = layer, calls
    def __call__(self, packed, bf16, meta):
        self.calls.append((self.layer, bf16.shape[0], meta.total_q))
        if isinstance(packed,tuple):
            data,sf=packed
            if data.shape[0] != 0 or tuple(sf.shape) != (0,40): raise ValueError("empty SF ABI")
        return bf16.clone()


class FakeLayer:
    def __init__(self, index, calls):
        self.layer_idx, self.mlp = index, FakeMLP(index,calls)
        self.inputs=[]
    def __call__(self, hidden, cache, meta, residual, *, embed_hook=None, **kwargs):
        self.inputs.append((hidden,meta,residual))
        if embed_hook is not None: embed_hook()
        return self.mlp(hidden,hidden,meta), residual


class FakeNativeModel:
    """Small real tensor loop fixture; kernels/projections are not simulated as passed."""
    def __init__(self, t):
        self.t=t;self.calls=[];self.fetches=[];self.native_calls=0
        self.layers=[FakeLayer(i,self.calls) for i in range(40)]
        self.ngram_multipliers=self.ngram_vocab_sizes=self.ngram_offsets=None
        def fetch(hashes):
            def hook(layer):
                self.fetches.append(layer)
                return t.zeros((1,5120),dtype=t.float32)
            return {1:lambda:hook(1),14:lambda:hook(14)}
        self.engram_runtime=NS(fetch=fetch)
    def embed_tokens(self, ids): return self.t.zeros((ids.numel(),5120),dtype=self.t.bfloat16)
    def allocate_output(self, n): return NS(hidden_states=self.t.zeros((n,5120),dtype=self.t.float32))
    def forward(self, caches, meta, hidden=None, dflash=None):
        self.native_calls+=1
        return self.allocate_output(1)
    def __call__(self,*args,**kwargs): return self.forward(*args,**kwargs)


def actual_native_vl_entry(model):
    """Execute frozen native Python entry methods without importing GPU modules.

    This checks the real wrapper ABI, not native kernels or GPU numerical parity.
    Only annotations are stripped; method bodies and super() dispatch are intact.
    """
    root=Path(__file__).resolve().parents[2]/"NativeSnapshot/deepinfer/model_executor/models"
    methods={}
    for filename,owner,names in (
        ("deepseek.py","DeepseekForCausalLM",("forward",)),
        ("deepseek_vl.py","DeepseekVLForCausalLM",
         ("_special_image_token_mask","get_input_embeddings","forward"))):
        parsed=ast.parse((root/filename).read_text())
        # The actual class spelling is taken from the pinned source. Do not
        # silently fall back to a hand-written wrapper if it changes.
        cls=next(node for node in parsed.body if isinstance(node,ast.ClassDef) and node.name==owner)
        body=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in body} != set(names): raise ValueError("native entry methods missing")
        for node in body:
            node.returns=None
            for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs): arg.annotation=None
        methods[filename]=body
    tree=ast.Module(body=[
        ast.ClassDef(name="NativeCausalEntry",bases=[],keywords=[],body=methods["deepseek.py"],decorator_list=[]),
        ast.ClassDef(name="NativeVLEntry",bases=[ast.Name(id="NativeCausalEntry",ctx=ast.Load())],
                     keywords=[],body=methods["deepseek_vl.py"],decorator_list=[])],type_ignores=[])
    ns={"DFlash":type("DFlash",(),{}),"ImageTokenOffset":NS(NUM_SPECIAL=4)}
    exec(compile(ast.fix_missing_locations(tree),str(root/"deepseek_vl.py"),"exec"),ns)
    entry=ns["NativeVLEntry"]();entry.model=model;entry.MTP_module=None;entry.cfg=NS(vocab_size=100)
    return entry.forward


def capability():
    # A spy of the exact required capability class, scoped solely to CPU tests.
    cap=object.__new__(EmptyEPCapability)
    def permits(layer,meta,plan):
        if meta.total_q!=1 or plan.full_rows!=1 or plan.selected_count!=0:
            raise ValueError("empty capability call geometry")
    cap.permits=permits
    return cap


def schedule(n=32769, group=2):
    return batch_schedule((dict(start=0,end=n,key="a"*64),),n,request_id="r",phase="p",
        phase_generation=2,input_ids_sha256="b"*64,group_chunks=group)


def world_samples(s, *, done=0, phase="prefill", ordinal=0):
    seqs = []
    if phase == "prefill":
        chunk=s["steps"][done]
        seqs=[dict(request_id=s["request_id"],is_prompt=True,
                   seqlen_q=chunk["end"]-chunk["start"],seqlen_k=chunk["end"])]
    elif phase == "decode":
        seqs=[dict(request_id=s["request_id"],is_prompt=False,
                   seqlen_q=1,seqlen_k=s["full_input_tokens"]+1)]
    return [dict(rank=rank,schedule_sha256=s["sha256"],ordinal=ordinal,prefill_completed=done,
                 prompt_run=phase=="prefill",is_empty=rank!=0 or not seqs,
                 sequences=seqs if rank==0 else []) for rank in range(8)]


class WorldSchedulerTests(unittest.TestCase):
    def test_idle_decode_before_first_and_between_prefill_does_not_advance(self):
        s=schedule()
        for done in (0,1,4):
            event=classify_world_step(world_samples(s,done=done,phase="idle"),s,0,done)
            self.assertEqual(event["world_phase"],"idle")
            self.assertEqual(event["prefill_completed_before"],done)
            self.assertIsNone(event["start"])

    def test_actual_request_decode_requires_all_prefill_chunks(self):
        s=schedule()
        with self.assertRaisesRegex(ValueError,"actual request decode"):
            classify_world_step(world_samples(s,phase="decode"),s,0,0)
        event=classify_world_step(world_samples(s,done=len(s["steps"]),phase="decode"),
                                  s,0,len(s["steps"]))
        self.assertEqual(event["world_phase"],"decode")

    def test_native_prefill_must_match_actual_absolute_chunk(self):
        s=schedule();samples=world_samples(s,done=1)
        samples[0]["sequences"][0]["seqlen_k"]-=1
        with self.assertRaisesRegex(ValueError,"actual prefill chunk"):
            classify_world_step(samples,s,0,1)

    def test_scope_progress_and_rank_order_cannot_drift(self):
        s=schedule()
        for key,value in (("rank",4),("prefill_completed",1),("ordinal",1),("schedule_sha256","x")):
            samples=world_samples(s);samples[2][key]=value
            with self.assertRaisesRegex(ValueError,"identity/progress"):
                classify_world_step(samples,s,0,0)

    def test_local_decode_is_not_relabelled_as_prompt(self):
        s=schedule();samples=world_samples(s);samples[1]["prompt_run"]=False
        with self.assertRaisesRegex(ValueError,"step types disagree"):
            classify_world_step(samples,s,0,0)

    def test_foreign_real_rank_or_request_is_rejected(self):
        s=schedule();samples=world_samples(s)
        samples[1].update(is_empty=False,sequences=samples[0]["sequences"])
        with self.assertRaisesRegex(ValueError,"inactive batch rank"):
            classify_world_step(samples,s,0,0)
        samples=world_samples(s);samples[0]["sequences"][0]["request_id"]="foreign"
        with self.assertRaisesRegex(ValueError,"request identity"):
            classify_world_step(samples,s,0,0)

    def test_rank0_engine_hook_records_gathered_evidence_and_preserves_return(self):
        s=schedule(n=16384);model=NS()
        controller=NS(native_model=model,state="RUNNING",batch_schedule=s,_batch_step_trace=[])
        class Engine:
            def _execute_model_async_step(self,seqs,outputs):
                if seqs and outputs.prompt_run:
                    step=s["steps"][len(controller._batch_step_trace)]
                    controller._batch_step_trace.append(dict(step,
                        deep_native_ep_calls=37 if step["flush"] else 0,empty_deep_native_ep_calls=0))
                return self.result
        engine=Engine();engine.result=object();samples=None
        def gather(value):
            self.assertEqual(value,samples[0])
            return samples
        worker=install_worker_batch_schedule(model,rank=0,torch_module=None,
            empty_ep_capability=capability(),controller=controller,engine=engine,phase_gather=gather)
        worker.configure(s)
        for phase,done in (("idle",0),("prefill",0),("idle",1),("prefill",1),("decode",2)):
            samples=world_samples(s,phase=phase,done=done,ordinal=len(worker.world_trace))
            out=engine._execute_model_async_step([NS(**x) for x in samples[0]["sequences"]],
                NS(prompt_run=samples[0]["prompt_run"],is_empty=lambda:samples[0]["is_empty"]))
            self.assertIs(out,engine.result)
        report=worker.finish_phase()
        self.assertEqual([e["world_phase"] for e in report["world_step_trace"]],
                         ["idle","prefill","idle","prefill","decode"])
        for event in report["world_step_trace"]:
            self.assertEqual(len(event["scheduler_observations"]),8)
        worker.close()
        self.assertNotIn("_execute_model_async_step",engine.__dict__)

    def test_engine_dependency_is_required_before_model_hook(self):
        model=FakeNativeModel(None)
        with self.assertRaisesRegex(ValueError,"actual engine"):
            install_worker_batch_schedule(model,rank=1,torch_module=None,empty_ep_capability=capability())
        self.assertNotIn("forward",model.__dict__)

    def test_foreign_engine_hook_is_rejected_before_model_installation(self):
        model=FakeNativeModel(None);engine=NS(_execute_model_async_step=lambda *a:None)
        with self.assertRaisesRegex(ValueError,"foreign engine"):
            install_worker_batch_schedule(model,rank=1,torch_module=None,
                empty_ep_capability=capability(),engine=engine,phase_gather=lambda x:[x])
        self.assertNotIn("forward",model.__dict__)


class SystemControllerContractTests(unittest.TestCase):
    def make(self):
        c=object.__new__(S1SystemController)
        c.state="IDLE";c._configured_system=False;c._configured_group=1
        c.system_qualified={1:False,2:False,4:False};c._empty_group_case=None
        c.cache=NS(configure=lambda **kw:None);c.attention=NS(configure=lambda **kw:None)
        return c

    def test_config_requires_explicit_bool_and_supported_width(self):
        c=self.make()
        for enabled,width in ((1,1),(True,3),(False,2)):
            with self.assertRaises(ValueError):c.configure(system_enabled=enabled,batch_group_chunks=width)
        c.configure(system_enabled=True,batch_group_chunks=4)
        self.assertEqual((c._configured_system,c._configured_group),(True,4))

    def test_active_request_cannot_switch_implementation(self):
        c=self.make();c.state="RUNNING"
        with self.assertRaises(ValueError):c.configure(system_enabled=True)

    def test_formal_optimized_requires_its_actual_group_qualification(self):
        c=self.make();c.configure(system_enabled=True,batch_group_chunks=4)
        c.system_qualified[2]=True
        with self.assertRaisesRegex(ValueError,"lacks actual"):
            c.reset("r","sensitive",verify_sparse=False)

    def test_reference_cannot_enable_new_policy_implementation(self):
        c=self.make();c.configure(system_enabled=True)
        with self.assertRaisesRegex(ValueError,"reference remains original"):
            c.reset("r","reference_sparse")

    def test_unbatched_model_delegates_to_original(self):
        c=self.make();c.batch_group_chunks=1
        with patch.object(S1Controller,"_policy_model_forward",return_value="original") as old:
            self.assertEqual(c._policy_model_forward(1,2,3,4,5),"original")
            old.assert_called_once_with(1,2,3,4,5)

    def test_formal_member_accounting_does_not_touch_gpu_tensors(self):
        c=self.make();c.mode="sensitive";c.verify_sparse=False;c.decoder_events=[]
        c.ctx=NS(positions_cpu=[8000,8200]);c.t=NS()
        c._active_deep_batch=NS(start=0,end=16384,member_chunks=[(0,8192),(8192,16384)],positions_cpu=[8000,8200])
        c._observe_layer(14,0,16384,(NS(shape=(2,5120)),object()),None)
        self.assertEqual([e["actual_ffn_rows"] for e in c.decoder_events],[1,1])
        self.assertEqual([e["native_ep_call_owner"] for e in c.decoder_events],[False,True])
        self.assertFalse(any(e["empty_native_ep_call"] for e in c.decoder_events))

    def test_empty_group_ep_is_counted_once_not_per_member(self):
        c=self.make();c.mode="sensitive";c.verify_sparse=False;c.decoder_events=[];c.ctx=NS();c.t=NS()
        c._active_deep_batch=NS(start=0,end=16384,member_chunks=[(0,8192),(8192,16384)],positions_cpu=[])
        c._observe_layer(3,0,16384,(NS(shape=(0,5120)),None),None)
        self.assertEqual(sum(e["empty_native_ep_call"] for e in c.decoder_events),1)
        self.assertEqual(sum(e["actual_attention_rows"] for e in c.decoder_events),0)

    def test_formal_member_split_detects_lost_actual_output(self):
        c=self.make();c.mode="sensitive";c.verify_sparse=False;c.decoder_events=[];c.ctx=NS();c.t=NS()
        c._active_deep_batch=NS(start=0,end=8192,member_chunks=[(0,8192)],positions_cpu=[8000])
        with self.assertRaisesRegex(ValueError,"not exhausted"):
            c._observe_layer(3,0,8192,(NS(shape=(0,5120)),None),None)

    def test_special_diagnostic_geometry_is_exact_and_not_general_shortcut(self):
        ids=list(range(32769));key=token_digest(ids)
        docs=[dict(input_start=0,input_end=32769,token_ids_sha256=key,document_id=key)]
        self.assertEqual(S1SystemController._empty_group_documents(ids,docs,"c"*64)[0]["end"],32769)
        with self.assertRaises(ValueError):S1SystemController._empty_group_documents(ids[:-1],docs,"c"*64)
        with self.assertRaises(ValueError):S1SystemController._empty_group_documents(ids,docs,"not-a-sha")

    def test_batch_cannot_execute_before_all_rank_bind(self):
        c=self.make();c.system_enabled=True;c.batch_group_chunks=2;c.mode="sensitive";c._batch=None
        with self.assertRaisesRegex(ValueError,"all-rank"):
            c._policy_model_forward(None,None,None,None,None)

    def test_capture_reset_clears_prior_system_cache_diagnostics(self):
        c=self.make();c.case_binding=None;c.phase="offline"
        c.cache.begin_phase=Mock()
        with patch.object(S1Controller,"reset",return_value=None):
            c.reset("r","capture_independent")
        c.cache.begin_phase.assert_called_once_with("offline",build_global_base=False)
        self.assertFalse(c.system_enabled)
        self.assertIsNone(c.batch_schedule)

    def test_missing_rank_acknowledgment_rejects_batch_before_forward(self):
        c=self.make();s=schedule();c.state="RUNNING";c.coverage=0;c._batch=None;c.batch_schedule=s
        with self.assertRaisesRegex(ValueError,"all-rank"):
            c.bind_batch_schedule(s,[])


class WorkerTensorMixin:
    def test_inactive_empty_cu_does_not_copy_host_list(self):
        meta=self.meta()
        host_method=(torch.Tensor if self.t is torch else RuntimeArray)
        with patch.object(host_method,'new_tensor',side_effect=AssertionError('host transfer forbidden')):
            actual=inactive_empty_query_cu(meta,self.t)
        self.assertEqual(actual.tolist(),[0,0])
        self.assertEqual(actual.dtype,meta.cu_seqlens_q.dtype)
        self.assertEqual(actual.device,meta.cu_seqlens_q.device)

    def meta(self):
        t=self.t
        return NS(is_prompt=True,num_reqs=1,total_q=1,canonical_request_binding=None,
            input_ids=t.tensor([0],dtype=t.int32),positions=t.tensor([0],dtype=t.int32),
            state_block_idx=t.tensor([0],dtype=t.int32),cu_seqlens_q=t.tensor([0,1],dtype=t.int32),
            ngram_token_ids=t.tensor([[0,0,0,0]],dtype=t.int32),
            mm_token_type=None,images_raw=None,images_seq_mask=None)
    def worker(self):
        model=FakeNativeModel(self.t)
        make_meta=self.meta
        class Engine:
            world_samples=None
            override_meta=None
            model_entry=None
            supplied_hidden=None
            def _execute_model_async_step(self,seqs,outputs):
                meta=self.override_meta or make_meta()
                meta.is_prompt=outputs.prompt_run
                return (self.model_entry or model.forward)([None]*40,meta,self.supplied_hidden)
        engine=Engine()
        def gather(sample):
            result=engine.world_samples
            # The tested worker supplies the same observation as the genuine
            # one-row empty native scheduler. Other ranks are CPU fixtures.
            if result[1]!=sample: raise ValueError("fixture scheduler observation differs")
            return result
        return model,install_worker_batch_schedule(model,rank=1,torch_module=self.t,
            empty_ep_capability=capability(),engine=engine,phase_gather=gather)

    def advance(self,worker,phase="prefill",meta=None):
        s=worker.schedule
        if s is None:
            worker.engine.override_meta=meta
            return worker.engine._execute_model_async_step([],NS(prompt_run=False,is_empty=lambda:True))
        samples=world_samples(s,done=len(worker.trace),phase=phase,ordinal=len(worker.world_trace))
        worker.engine.world_samples=samples;worker.engine.override_meta=meta
        return worker.engine._execute_model_async_step([],NS(prompt_run=phase=="prefill",is_empty=lambda:True))

    def test_worker_native_first3_every_step_deep_only_group2_flush(self):
        model,worker=self.worker();s=schedule()
        receipt=worker.configure(s);self.assertTrue(receipt["worker_loop_installed"])
        for _ in s["steps"]:
            output=self.advance(worker)
            self.assertEqual(tuple(output.hidden_states.shape),(1,5120))
        result=worker.finish_phase()
        self.assertEqual(result["actual_deep_ep_calls"],3*37)
        self.assertEqual(model.fetches,[1,14]*5)
        self.assertEqual([x[0] for x in model.calls],sum([step["layers"] for step in s["steps"]],[]))
        self.assertTrue(all(rows==meta_rows==(1 if layer<3 else 0) for layer,rows,meta_rows in model.calls))
        self.assertEqual([e["world_phase"] for e in result["world_step_trace"]],["prefill"]*5)
        worker.close();self.assertNotIn("forward",model.__dict__)
        self.assertNotIn("_execute_model_async_step",worker.engine.__dict__)

    def test_worker_group4_flush_and_partial_last_group(self):
        model,worker=self.worker();s=schedule(group=4);worker.configure(s)
        for _ in s["steps"]:self.advance(worker)
        self.assertEqual(worker.finish_phase()["actual_deep_ep_calls"],74)
        worker.close()

    def test_actual_native_vl_prefill_preserves_embedding_and_image_mask_through_flush(self):
        model,worker=self.worker();s=schedule();worker.configure(s)
        worker.engine.model_entry=actual_native_vl_entry(model)
        original=model.embed_tokens;values=[]
        def embed(ids):
            value=original(ids);value[:]=3
            values.append(value)
            return value
        with patch.object(model,"embed_tokens",side_effect=embed) as call:
            for _ in s["steps"]:self.advance(worker)
        self.assertEqual(call.call_count,len(s["steps"]))
        self.assertEqual(len(model.layers[0].inputs),len(values))
        for value,(seen,meta,residual) in zip(values,model.layers[0].inputs):
            self.assertIs(seen,value)
            self.assertIsNotNone(meta.images_seq_mask)
            self.assertFalse(bool(meta.images_seq_mask[0]))
            self.assertIsNone(residual)
        self.assertEqual(worker.finish_phase()["actual_deep_ep_calls"],111)
        worker.close()

    def test_supplied_embedding_is_used_without_embedding_again(self):
        model,worker=self.worker();worker.configure(schedule())
        supplied=self.t.zeros((1,5120),dtype=self.t.bfloat16)
        supplied[:]=7;worker.engine.supplied_hidden=supplied
        with patch.object(model,"embed_tokens",side_effect=AssertionError("must preserve input")):
            self.advance(worker)
        self.assertIs(model.layers[0].inputs[0][0],supplied)
        worker.close()

    def test_supplied_embedding_invalid_shape_or_dtype_is_rejected_before_layers(self):
        for shape,dtype in (((2,5120),self.t.bfloat16),((1,5119),self.t.bfloat16),
                            ((1,5120),self.t.float64)):
            model,worker=self.worker();worker.configure(schedule())
            worker.engine.supplied_hidden=self.t.zeros(shape,dtype=dtype)
            with self.assertRaisesRegex(ValueError,"native input embedding"):
                self.advance(worker)
            self.assertFalse(model.calls);worker.close()

    def test_worker_disabled_uses_original_with_no_new_ep(self):
        model,worker=self.worker();worker.configure(None)
        self.advance(worker)
        self.assertEqual(model.native_calls,1);self.assertFalse(model.calls)
        self.assertFalse(worker.finish_phase()["enabled"]);worker.close()

    def test_inactive_worker_rejects_real_request_state(self):
        model,worker=self.worker();worker.configure(schedule());meta=self.meta();meta.state_block_idx[0]=1
        with self.assertRaises(Exception):self.advance(worker,meta=meta)
        self.assertFalse(model.calls);worker.close()

    def test_actual_early_decode_is_rejected_before_any_model_work(self):
        model,worker=self.worker();worker.configure(schedule())
        with self.assertRaisesRegex(ValueError,"actual request decode"):
            self.advance(worker,phase="decode")
        self.assertFalse(model.calls);self.assertEqual(model.native_calls,0)
        with self.assertRaises(ValueError):worker.finish_phase()
        worker.close()

    def test_real_native_idle_decode_before_between_and_after_prompt(self):
        model,worker=self.worker();s=schedule();worker.configure(s)
        self.advance(worker,phase="idle")
        self.assertEqual(worker.trace,[])
        for _ in s["steps"]:
            self.advance(worker)
            self.advance(worker,phase="idle")
        self.advance(worker,phase="decode")
        self.assertEqual(model.native_calls,7)
        self.assertEqual(len(worker.trace),5)
        report=worker.finish_phase()
        self.assertEqual(report["actual_deep_ep_calls"],3*37)
        self.assertEqual(sum(e["world_phase"]=="idle" for e in report["world_step_trace"]),6)
        worker.close()

    def test_direct_model_cannot_consume_unbound_schedule(self):
        model,worker=self.worker();worker.configure(schedule())
        with self.assertRaisesRegex(ValueError,"world scheduler authorization"):
            model.forward([],self.meta())
        self.assertFalse(model.calls);worker.close()

    def test_native_model_exception_clears_step_authorization(self):
        model,worker=self.worker();worker.configure(schedule())
        with patch.object(model,"embed_tokens",side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError,"fixture failure"):self.advance(worker)
        self.assertIsNone(worker._world_frame)
        self.assertFalse(worker.world_trace);worker.close()

    def test_worker_foreign_hook_is_not_removed(self):
        model,worker=self.worker();model.forward=lambda *a:None
        with self.assertRaisesRegex(ValueError,"ownership"):worker.close()


@unittest.skipIf(np is None,"NumPy CPU fixture required")
class WorkerArrayTests(WorkerTensorMixin,unittest.TestCase):
    def setUp(self):self.t=RuntimeNumpy


@unittest.skipIf(torch is None,"Torch CPU required; remote admission forbids skips")
class WorkerTorchTests(WorkerTensorMixin,unittest.TestCase):
    def setUp(self):self.t=torch


if __name__=="__main__":unittest.main()
