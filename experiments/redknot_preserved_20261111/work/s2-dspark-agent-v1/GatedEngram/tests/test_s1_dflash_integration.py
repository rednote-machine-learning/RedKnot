"""CPU-only contracts for actual new-stage bridge/decoder/worker methods."""
import hashlib
import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from kv_cache_engram.s1_runtime import S1Controller
from kv_cache_engram.s1_dflash_bridge import S1DFlashBridge
from tests.test_s1_system_runtime import (WorkerTensorMixin, RuntimeNumpy, schedule, world_samples)


class Tensor:
    def __init__(self, data, dtype="bfloat16", device="cpu-fixture"):
        self.a, self.dtype, self.device = np.asarray(data), dtype, device
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    def is_contiguous(self): return self.a.flags.c_contiguous
    def new_empty(self, shape): return Tensor(np.empty(shape, np.float32), self.dtype, self.device)
    def view(self, *shape): return Tensor(self.a.reshape(shape), self.dtype, self.device)
    def narrow(self, dim, start, length):
        ix=[slice(None)]*self.ndim;ix[dim]=slice(start,start+length)
        return Tensor(self.a[tuple(ix)],self.dtype,self.device)
    def to(self, *, dtype): return Tensor(self.a,dtype,self.device)
    def __getitem__(self, ix): return Tensor(self.a[ix],self.dtype,self.device)
    def __mod__(self, x): return Tensor(self.a%x,self.dtype,self.device)
    def __floordiv__(self, x): return Tensor(self.a//x,self.dtype,self.device)
    def __sub__(self, x): return Tensor(self.a-x,self.dtype,self.device)
    def __add__(self, x): return Tensor(self.a+(x.a if isinstance(x,Tensor) else x),self.dtype,self.device)
    __radd__=__add__
    def __eq__(self,x): return self.a==(x.a if isinstance(x,Tensor) else x)
    def __gt__(self,x): return self.a>x
    def __ge__(self,x): return self.a>=x
    def __lt__(self,x): return self.a<x
    def item(self): return self.a.item()
    def index_select(self,dim,indices): return Tensor(np.take(self.a,indices.a.astype(int),axis=dim),self.dtype,self.device)


class Torch:
    bfloat16="bfloat16";long="int64"
    @staticmethod
    def arange(a,b,*,dtype,device):return Tensor(np.arange(a,b),dtype,device)
    @staticmethod
    def _assert_async(ok,message):
        if not bool(ok):raise ValueError(message)


def native_store(hidden,out):out.a[...]=hidden.a.mean(axis=1)
def fingerprint(t,unused):
    return dict(shape=list(t.shape),dtype=t.dtype,sha256=hashlib.sha256(t.a.tobytes()).hexdigest())
def features(positions,layer):
    return Tensor(np.array([[[p*10+layer+h+d for d in range(4)] for h in range(4)]
                            for p in positions],dtype=np.float32)[None])


class Layer:
    def __init__(self,i,index):self.layer_idx,self.dflash_input_index=i,index;self.native_stores=[]
    def _store_dflash_input(self,hidden,out):self.native_stores.append((hidden,out))
    def forward(self,hidden,cache,meta,residual,**kwargs):
        self._store_dflash_input(residual.residual,kwargs["dflash_input"])
        return hidden,residual


class BridgeTests(unittest.TestCase):
    def owner(self):
        layers=[Layer(i,{20:0,2:1}.get(i)) for i in range(40)]
        o=NS(layers=layers,native_model=NS(dflash_input_size=8,hidden_size=4),t=Torch,
             mode="reference_sparse",rid="r",phase="p",ids=tuple(range(260)),input_sha="a"*64,
             prepared={(0,260):dict(plan=dict(selected_positions=list(range(132,260))))},
             handles=[],coverage=0,ctx=None,allow_sync=nullcontext,verify_sparse=False)
        o._patch=lambda obj,name,fn:S1Controller._patch(o,obj,name,fn)
        o._marker=lambda marker:(marker["start"],marker["end"])
        o._observe_layer=lambda *args:None
        o._qualifying_sparse=lambda:o.mode=="reference_sparse" or o.verify_sparse
        o.dflash_bridge=S1DFlashBridge(o,native_input_store=native_store).install()
        o.dflash_bridge.reset()
        return o

    def run_layer(self,o,layer,positions,*,start,end,native_output=None):
        meta=NS(is_prompt=True,canonical_request_binding=dict(start=start,end=end))
        h=features(positions,layer)
        o.layers[layer].forward(Tensor(np.zeros((len(positions),4))),None,meta,NS(residual=h),dflash_input=native_output)
        return h

    def final(self,o):
        calls=[];ring=134;state=2;n=len(o.ids)
        meta=NS(is_prompt=True,num_reqs=1,mirror_cp_reverse_index=None,
                canonical_request_binding=dict(request_id=o.rid,phase=o.phase,end=n),
                positions=Tensor([n-1],"int64"),state_block_idx=Tensor([state],"int32"),
                slot_mapping_store_swa=Tensor([state*ring+(n-1)%ring],"int32"))
        caches={40:NS(swa_cache=Tensor(np.zeros((4,ring,1,8),np.float32)))}
        def update(buf,m,caches):
            calls.append(buf.a.copy())
            caches[40].swa_cache.a[state,m.positions.a%ring,0,:]=buf.a
        dflash=NS(target_layer_ids=(20,2),window_size=128,layers=[NS(layer_idx=40)],update_main_cache=update)
        return meta,dflash,caches,calls

    def test_actual_decoder_routes_first3_and_grouped_selected_rows(self):
        o=self.owner();o.mode="sensitive"
        for i in (2,20):S1Controller._install_decoder(o,o.layers[i])
        self.run_layer(o,2,list(range(200)),start=0,end=200)
        self.run_layer(o,2,list(range(200,260)),start=200,end=260)
        o.ctx=NS(positions_cpu=list(range(132,259)))
        h=self.run_layer(o,20,list(range(132,259)),start=0,end=259)
        h.a.fill(-999)
        o.ctx=NS(positions_cpu=[259]);self.run_layer(o,20,[259],start=259,end=260)
        meta,dflash,caches,calls=self.final(o)
        o.dflash_bridge.finish(meta,dflash,caches)
        r=o.dflash_bridge.report()
        self.assertEqual(r["retained_positions"],list(range(132,260)))
        self.assertEqual(r["native_main_cache_update_calls"],1)
        self.assertEqual(r["coverage_by_target"],{"20":128,"2":128})
        self.assertFalse(r["sampler_placeholder_features_used"])
        self.assertIsNone(r["qualification_fingerprints"])
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0,0],1320+20+1.5)
        self.assertEqual(calls[0][-1,-1],2590+2+1.5+3)

    def test_real_reference_candidate_feature_and_cache_hashes_are_compared(self):
        o=self.owner();b=o.dflash_bridge
        def commit(delta=0):
            for i in (20,2):
                with b.layer_call(i,list(range(132,260))):
                    h=features(range(132,260),i);h.a+=delta
                    o.layers[i]._store_dflash_input(h,None)
            meta,native,caches,calls=self.final(o)
            with patch("kv_cache_engram.s1_dflash_bridge.fingerprint",fingerprint):b.finish(meta,native,caches)
            return b.report()
        ref=commit()
        o.mode="sensitive";o.verify_sparse=True;b.reset();got=commit()
        self.assertEqual(ref["qualification_fingerprints"],got["qualification_fingerprints"])
        self.assertTrue(got["same_policy_reference_exact"]);self.assertEqual(got["state_id"],2)
        b.reset()
        with self.assertRaisesRegex(ValueError,"byte parity"):commit(delta=1)

    def test_original_native_stores_survive_all_other_modes_and_decode(self):
        o=self.owner();layer=o.layers[20]
        S1Controller._install_decoder(o,layer)
        for mode in ("native","qualified_native","all_selected","capture_independent"):
            o.mode=mode;payload=object()
            meta=NS(is_prompt=False,canonical_request_binding=None)
            layer.forward(None,None,meta,NS(residual=payload),dflash_input=payload)
        self.assertEqual(len(layer.native_stores),4)
        o.mode="native"
        with self.assertRaisesRegex(ValueError,"owned policy"):layer._store_dflash_input(None,None)

    def test_reference_decoder_uses_dense_positions_before_gather(self):
        o=self.owner();layer=o.layers[20];S1Controller._install_decoder(o,layer)
        o.ctx=NS(positions_cpu=list(range(132,260)))
        self.run_layer(o,20,list(range(260)),start=0,end=260)
        self.assertEqual(o.dflash_bridge.collector.coverage[20],set(range(132,260)))

    def test_no_target_feature_or_missing_final_tail_fails_closed(self):
        o=self.owner();meta,dflash,caches,calls=self.final(o)
        with self.assertRaisesRegex(ValueError,"incomplete"):o.dflash_bridge.finish(meta,dflash,caches)
        self.assertEqual(calls,[])
        with self.assertRaisesRegex(ValueError,"did not initialize"):o.dflash_bridge.report()

    def test_new_request_clears_features_but_only_explicit_bank_clear_resets_reference(self):
        o=self.owner();b=o.dflash_bridge;b.reference=("bound",);o.mode="sensitive"
        b.reset();self.assertIsNone(b.collector.buffer);self.assertEqual(b.reference,("bound",))
        o.mode="reference_sparse";b.reset();self.assertIsNone(b.reference)


class NativeWorkerGraphTests(unittest.TestCase):
    def fixture(self):
        f=WorkerTensorMixin();f.t=RuntimeNumpy
        model,worker=f.worker();worker.configure(schedule())
        return f,model,worker

    def test_native_idle_graph_can_omit_python_model_callback_only_with_optin(self):
        f,model,w=self.fixture();w.allow_native_decode_graphs=True
        w.engine_original=lambda seqs,outputs: "native-graph-result"
        self.assertEqual(f.advance(w,phase="idle"),"native-graph-result")
        self.assertEqual(w.trace,[])
        self.assertFalse(w.world_trace[0]["python_model_callback_consumed"])
        w.close()
        f,model,w=self.fixture();w.engine_original=lambda seqs,outputs:None
        with self.assertRaisesRegex(ValueError,"bypassed"):f.advance(w,phase="idle")
        w.close()

    def test_prefill_must_still_execute_owned_callback_under_graph_optin(self):
        f,model,w=self.fixture();w.allow_native_decode_graphs=True
        w.engine_original=lambda seqs,outputs:None
        with self.assertRaisesRegex(ValueError,"follow world"):f.advance(w)
        w.close()

    def test_native_idle_eager_path_remains_original_with_dflash_optin(self):
        f,model,w=self.fixture();w.allow_native_decode_graphs=True
        f.advance(w,phase="idle")
        self.assertEqual(model.native_calls,1);self.assertEqual(w.trace,[])
        w.close()

    def test_inactive_prefill_has_native_target_buffer_but_no_real_tail_commit(self):
        f,model,w=self.fixture();w.allow_native_decode_graphs=True;model.dflash_input_size=12
        dflash=object();buffers=[]
        original=model.layers[2].__class__.__call__
        def layer_call(layer,*args,**kwargs):
            buffers.append(kwargs.get("dflash_input"));return original(layer,*args,**kwargs)
        def native_engine(seqs,outputs):return model.forward([None]*40,f.meta(),None,dflash)
        w.engine_original=native_engine
        with patch.object(model.layers[2].__class__,"__call__",layer_call):f.advance(w)
        self.assertEqual(len(buffers),3)
        self.assertTrue(all(x.shape==(1,12) for x in buffers))
        self.assertEqual(model.native_calls,0);self.assertEqual(len(w.trace),1)
        w.close()


if __name__=="__main__":unittest.main()
