"""CPU contract tests; real native numerical qualification remains mandatory."""
import ast
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.ffn_only import FFNOnlyController
from kv_cache_engram.swa_reuse_qonly import SwaReuseQOnlyController, POLICY
from tests.test_ffn_only_controller import Adapter


def controller():
    layers = tuple(SimpleNamespace(layer_idx=i) for i in range(40))
    c = SwaReuseQOnlyController(layers, model_sha256="a"*64, execution_sha256="b"*64,
        tokenizer_sha256="c"*64, native_model=SimpleNamespace(layers=layers,yoco_boundary_layer_idx=20))
    c.original_yoco_boundary, c.adapter, c.state = 20, Adapter(), "IDLE"
    return c


class Bits:
    def __init__(self, values):
        self.values = list(values)
    def fill_(self, value):
        self.values[:] = [value]*len(self.values)
        return self
    def index_fill_(self, dim, ids, value):
        assert dim == 0
        for i in ids:
            self.values[i] = value
        return self


class PolicyTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_native(self):
        result = subprocess.run([sys.executable,"-B","-c",
            "import sys; import kv_cache_engram.swa_reuse_qonly; assert 'torch' not in sys.modules; assert not any(n.startswith('deepinfer') for n in sys.modules)"],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cached_first127_and_last_rows_are_not_online(self):
        c = controller(); c.mode = "sensitive"
        mandatory = [0,1,998,999]
        optional = list(range(2,998))
        c.prepared = {(0,1000):dict(mandatory=mandatory,optional=optional,online=list(range(129))+[998,999],online_mask=Bits([True]*1000))}
        c._configure_document_reuse()
        self.assertIs(c.prepared[(0,1000)]["online"],mandatory)
        self.assertIs(c.prepared[(0,1000)]["optional"],optional)
        self.assertEqual([i for i,v in enumerate(c.prepared[(0,1000)]["online_mask"].values) if v],mandatory)

    def test_full_document_chunk_has_zero_online_kv_rows(self):
        c = controller(); c.mode = "sensitive"
        c.prepared = {(0,8192):dict(mandatory=[],optional=list(range(8192)),online=list(range(127)),online_mask=Bits([True]*8192))}
        c._configure_document_reuse()
        self.assertEqual(c.prepared[(0,8192)]["online"],[])
        self.assertFalse(any(c.prepared[(0,8192)]["online_mask"].values))

    def test_all_selected_qualification_policy_not_changed(self):
        c = controller(); c.mode = "all_selected"
        old = dict(mandatory=[0],online=[0,1],online_mask=Bits([True,True]))
        c.prepared = {(0,2):old}
        c._configure_document_reuse()
        self.assertEqual(old["online"],[0,1])

    def test_ffn_selection_and_decode_implementations_are_inherited(self):
        for name in ("_chunk_plan","get_ffn_plan","_selected_metadata","_layout","_make_projections"):
            self.assertIs(getattr(SwaReuseQOnlyController,name),getattr(FFNOnlyController,name))
        source = inspect.getsource(SwaReuseQOnlyController._install_attention)
        self.assertNotIn('"_decode_attention"',source)
        self.assertNotIn("use_swa =",source)
        self.assertNotIn("use_swa=False",source)
        self.assertIn("transparent_swa_binding",source)

    def test_policy_reconfiguration_adds_no_host_sync_or_tensor_factory(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(SwaReuseQOnlyController._configure_document_reuse)))
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & {"cpu","tolist","numpy","tensor","item","nonzero","index_put_"})

    def test_missing_actual_joint_attention_fails_after_parent_finish(self):
        c=controller(); c.mode="sensitive"
        report=dict(swa_events=[dict(layer=6,chunk_start=0,chunk_end=8,full_rows=8)])
        with patch.object(FFNOnlyController,"finish",return_value=report),self.assertRaises(ValidationError):
            c.finish()
        self.assertEqual(c.state,"FAILED")

    def test_report_native_case_keeps_policy_and_no_fabricated_cache_usage(self):
        c=controller(); c.mode="native"
        with patch.object(FFNOnlyController,"finish",return_value=dict(swa_events=[])):
            report=c.finish()
        self.assertEqual(report["swa_reuse_policy"],POLICY)
        self.assertEqual(report["effective_cache_attention_rows"],0)
        self.assertTrue(report["joint_attention_native"])
        self.assertTrue(report["decode_native"])

    def test_actual_finish_schema_passes_independent_controller_reader(self):
        from tests.test_swa_reuse_qonly_pilot import fixture
        from gpu_pilot.read_swa_reuse_qonly_results import audit_controller
        base,row,phase,spans=fixture()
        c=controller(); c.mode="sensitive"
        c.joint_attention_events=base["joint_attention_events"]
        c.canonical_copy_events=base["canonical_copy_events"]
        with patch.object(FFNOnlyController,"finish",return_value=base):
            report=c.finish()
        result=audit_controller(report,row,phase,spans)
        self.assertEqual(result["effective_cache_attention_rows"],7192)


TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH,"Torch CPU runtime unavailable")
class TorchPathTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def small(self, *, mode="sensitive", docs=((2,6),), positions=None):
        t,c=self.t,controller()
        c.t,c.state,c.mode=t,"RUNNING",mode
        c.rid,c.phase,c.ids,c.input_sha="r","p",(1,)*8,"d"*64
        c.docs=tuple(dict(start=a,end=b,key="e"*64) for a,b in docs)
        c.scores,c.plans,c.layouts,c.ffn_layouts,c.swa_events={},{},{},[],[]
        c.projection_events,c.projection_checks=[],0
        c.ops=SimpleNamespace(PG=SimpleNamespace(context_parallel_group=lambda:SimpleNamespace(size=lambda:1)),
            rows=lambda x:x.shape[0],normalize=lambda mod,x:x,
            postprocess=lambda mod,x,meta:(x.reshape(meta.total_q,1,512),None),
            identity_error=lambda a,b: self.assertTrue(t.equal(a,b)))
        optional=[i for i in range(8) if any(a<=i<b for a,b in docs)]
        mandatory=[i for i in range(8) if i not in optional]
        c.prepared={(0,8):dict(optional=t.tensor(optional,dtype=t.long),mandatory=t.tensor(mandatory,dtype=t.long),
            online=t.arange(8),online_mask=t.ones(8,dtype=t.bool))}
        c._configure_document_reuse()
        m=dict(request_id="r",phase="p",start=0,end=8)
        pos=t.arange(8) if positions is None else t.tensor(positions,dtype=t.long)
        meta=SimpleNamespace(is_prompt=True,total_q=pos.numel(),positions=pos,canonical_request_binding=m,
            kv_swa_full_index=object(),total_k_compressed={"r":2})
        for layer in (2,3,4,5):
            c._score(layer,m,t.ones(8,2))
        c._entry=Mock(return_value={"tensor":t.full((8,512),7.0)})
        return c,meta

    def attention(self,c):
        t=self.t
        class Attention:
            layer_idx=6
            use_swa=has_packed_kv=True
            is_kv_source=False
            use_shared_kv=True
            ratio_key="r"
            yoco_swa_share_hidden=False
            def __init__(a): a.calls=[]
            def _project_q_and_swa(a,h,s):
                a.calls.append(("native-project",h.shape[0]))
                return t.zeros(h.shape[0],1280,dtype=t.bfloat16),t.ones(h.shape[0],512,dtype=t.bfloat16)
            def _prepare_swa_kv(a,raw,cache,meta,use_cp):
                a.calls.append(("native-prepare",raw.shape[0]))
                return raw.reshape(meta.total_q,1,512),None
            def _compute_kv_ca_index(a,*args):
                a.calls.append(("global-compute",))
                return object()
            def _prefill_attention(a,q,kv,g,s,past,current,meta):
                a.calls.append(("native-joint",kv.clone(),s,g))
                return kv
            def forward(a,h,cache,meta,bf16,cross_layer_runtime=None):
                q,raw=a._project_q_and_swa(h,None)
                swa,_=a._prepare_swa_kv(raw,cache,meta,False)
                g=a._compute_kv_ca_index(None,None,None,cache,meta,cross_layer_runtime)
                kv=t.cat((swa,t.full((2,1,512),9.0)),0)
                return a._prefill_attention(q,kv,g,meta.kv_swa_full_index,0,meta.total_q,meta)
        a=Attention()
        def q_proj(h):
            a.calls.append(("q-only",h.shape[0]))
            return t.zeros(h.shape[0],1280,dtype=t.bfloat16)
        def kv_proj(h):
            a.calls.append(("kv-selected",h.shape[0]))
            return t.ones(h.shape[0],512,dtype=t.bfloat16)
        c.projections[6]=(q_proj,kv_proj)
        c._install_attention(a)
        return a

    def run_attention(self,c,meta):
        a=self.attention(c)
        with patch.object(self.t.cuda,"is_current_stream_capturing",return_value=False):
            result=a.forward(self.t.zeros(meta.total_q,5120),None,meta,self.t.zeros(meta.total_q,5120))
        return a,result

    def test_fully_cached_chunk_really_skips_swa_projection_but_runs_joint(self):
        c,m=self.small(docs=((0,8),))
        a,result=self.run_attention(c,m)
        self.assertFalse(any(x[0] in ("kv-selected","native-project","native-prepare") for x in a.calls))
        self.assertTrue(any(x[0]=="q-only" for x in a.calls))
        self.assertTrue(self.t.equal(result[:8],self.t.full((8,1,512),7.0)))
        self.assertTrue(self.t.equal(result[8:],self.t.full((2,1,512),9.0)))
        self.assertEqual(c.joint_attention_events[0]["reused_swa_rows"],8)
        self.assertEqual(c.canonical_copy_events[0]["canonical_bf16_bytes"],8*512*2)

    def test_mixed_rows_only_project_non_document_kv(self):
        c,m=self.small()
        a,result=self.run_attention(c,m)
        self.assertIn(("kv-selected",4),a.calls)
        self.assertEqual(c.swa_events[0]["online_local_rows"].tolist(),[0,1,6,7])
        self.assertTrue(self.t.equal(result[2:6],self.t.full((4,1,512),7.0)))
        self.assertTrue(self.t.equal(result[[0,1,6,7]],self.t.ones(4,1,512)))

    def test_compact_yoco_intersects_cache_without_forcing_ffn_or_boundary(self):
        c,m=self.small(positions=[2,4,6])
        self.assertEqual(c._layout(m)["online_ids"].tolist(),[2])
        plan=c.get_ffn_plan(SimpleNamespace(layer_idx=39),m,None)
        self.assertEqual(plan.selected_rows.tolist(),[0,2])
        a,result=self.run_attention(c,m)
        self.assertIn(("kv-selected",1),a.calls)
        self.assertEqual(c.joint_attention_events[0]["reused_swa_rows"],2)

    def test_all_selected_keeps_native_result_and_two_restore_identity_checks(self):
        c,m=self.small(mode="all_selected",docs=())
        a,result=self.run_attention(c,m)
        self.assertEqual(c.projection_checks,3)
        self.assertEqual(c.cache_restore_identity_events[0]["checks"],2)
        self.assertEqual(c.cache_restore_identity_events[0]["max_absolute_error"],0.0)
        self.assertEqual(c.joint_attention_events[0]["reused_swa_rows"],0)
        self.assertTrue(self.t.equal(result[:8],self.t.ones(8,1,512)))
        from gpu_pilot.read_swa_reuse_qonly_results import cache_restore_identity,audit_joint_attention
        swa=[{key:(value.tolist() if isinstance(value,self.t.Tensor) else value) for key,value in e.items()} for e in c.swa_events]
        with patch.object(FFNOnlyController,"finish",return_value=dict(swa_events=swa)):
            report=c.finish()
        self.assertEqual(cache_restore_identity(report),2)
        self.assertEqual(audit_joint_attention(report),0)

    def test_cache_restore_mismatch_aborts_qualification(self):
        c,m=self.small(mode="all_selected",docs=())
        c.ops.postprocess=lambda mod,x,meta:(self.t.zeros(meta.total_q,1,512),None)
        with self.assertRaises(AssertionError):
            self.run_attention(c,m)
        self.assertEqual(c.state,"FAILED")

    def test_foreign_native_cache_session_rejected_before_computation(self):
        c,m=self.small()
        m.transparent_swa_binding={"session":object()}
        a=self.attention(c)
        with self.assertRaises(ValidationError):
            a.forward(None,None,m,None)
        self.assertEqual(a.calls,[])

    def test_compact_zero_rows_preserve_native_joint_call(self):
        c,m=self.small(positions=[])
        a,result=self.run_attention(c,m)
        self.assertEqual(c.joint_attention_events[0]["full_rows"],0)
        self.assertEqual(result.shape,(2,1,512))


if __name__=="__main__":
    unittest.main()
