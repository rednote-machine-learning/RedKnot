"""Standard-library CPU fixtures, not a model accuracy/calibration result."""
from dataclasses import FrozenInstanceError, replace
import math
import subprocess
import sys
import unittest

from kv_cache_engram.core import Binding, CacheIdentity, OnlineRows, ValidationError
from kv_cache_engram.gate import BlockPlan, GateDecision, GatePolicy, evaluate, make_plan


def identity(token_count=2048, width=2):
    return CacheIdentity(model_sha256='a'*64, tokenizer_sha256='b'*64,
        execution_sha256='c'*64, tokens_sha256='d'*64, layer=3,
        token_count=token_count, width=width)


def binding(start=512, end=1024, token_count=2048, width=2, **updates):
    values=dict(request_id='request-a', phase='test', identity=identity(token_count, width),
                document_start=19, block_start=start, block_end=end, request_sha256='e'*64)
    values.update(updates)
    return Binding(**values)


def policy(**updates):
    values=dict(enabled=True, relative_tolerance=.1, absolute_tolerance=.2,
                calibration_id='synthetic-test-fixture-not-real-calibration')
    values.update(updates)
    return GatePolicy(**values)


def observation(bound, plan, row=(1., 2.)):
    return OnlineRows(binding_sha256=bound.sha256, positions=plan.probe_positions,
                      rows=tuple(tuple(row) for _ in plan.probe_positions))


class PolicyTests(unittest.TestCase):
    def test_defaults_never_enable_reuse_or_invent_tolerances(self):
        config=GatePolicy()
        self.assertFalse(config.enabled)
        self.assertIsNone(config.relative_tolerance)
        self.assertIsNone(config.absolute_tolerance)
        self.assertIsNone(config.calibration_id)
        bound=binding(); plan=make_plan(bound,config)
        result=evaluate(plan,config,None,None,bound)
        self.assertEqual(result,GateDecision(False,'policy_disabled',None,None))

    def test_enabled_requires_both_limits_and_calibration_declaration(self):
        for changes in ({'relative_tolerance':None},{'absolute_tolerance':None},
                        {'calibration_id':None},{'calibration_id':''},{'calibration_id':' '}):
            with self.subTest(changes=changes),self.assertRaises(ValidationError):
                policy(**changes)
        self.assertTrue(policy(relative_tolerance=0.,absolute_tolerance=0.).enabled)

    def test_rejects_nonfinite_negative_bool_and_invalid_sizes(self):
        for name in ('relative_tolerance','absolute_tolerance'):
            for bad in (-1.,float('nan'),float('inf'),True,'1',10**1000):
                with self.subTest(name=name,bad=bad),self.assertRaises(ValidationError):policy(**{name:bad})
        for name in ('block_size','probe_count'):
            for bad in (0,-1,True,1.5):
                with self.subTest(name=name,bad=bad),self.assertRaises(ValidationError):policy(**{name:bad})
        for name in ('head_guard','tail_guard'):
            for bad in (-1,True,1.5):
                with self.subTest(name=name,bad=bad),self.assertRaises(ValidationError):policy(**{name:bad})
        for bad in (0.,-1.,float('nan'),float('inf'),True):
            with self.subTest(norm_floor=bad),self.assertRaises(ValidationError):policy(norm_floor=bad)
        for changes in ({'seed':''},{'seed':' '},{'seed':None},{'enabled':1}):
            with self.subTest(changes=changes),self.assertRaises(ValidationError):policy(**changes)

    def test_policy_and_plan_are_immutable(self):
        config=policy();plan=make_plan(binding(),config)
        with self.assertRaises(FrozenInstanceError):config.enabled=False
        with self.assertRaises(FrozenInstanceError):plan.positions=()


class PlanningTests(unittest.TestCase):
    def test_guards_only_apply_at_document_boundaries_not_every_block(self):
        config=policy()
        first=make_plan(binding(0,512),config)
        middle=make_plan(binding(512,1024),config)
        last=make_plan(binding(1536,2048),config)
        self.assertEqual(first.mandatory_positions,tuple(range(128)))
        self.assertEqual(middle.mandatory_positions,())
        self.assertEqual(last.mandatory_positions,tuple(range(1920,2048)))
        self.assertTrue(set(first.probe_positions).isdisjoint(first.mandatory_positions))
        self.assertTrue(set(last.probe_positions).isdisjoint(last.mandatory_positions))

    def test_shifted_document_keeps_document_local_guard_positions(self):
        config=policy();bound=binding(0,512,document_start=32787)
        plan=make_plan(bound,config)
        self.assertEqual(plan.positions,tuple(range(512)))
        self.assertEqual(plan.mandatory_positions,tuple(range(128)))

    def test_tail_block_and_overlapping_short_document_guards(self):
        config=policy()
        tail=make_plan(binding(1024,1100,token_count=1100),config)
        self.assertEqual(tail.mandatory_positions,tail.positions)
        self.assertEqual(tail.probe_positions,())
        short_binding=binding(0,200,token_count=200)
        short=make_plan(short_binding,config)
        self.assertEqual(short.mandatory_positions,tuple(range(200)))
        self.assertEqual(evaluate(short,config,None,None,short_binding).reason,'no_reusable_positions')

    def test_alignment_and_full_block_required_except_document_tail(self):
        for bound in (binding(1,513),binding(0,400),binding(512,900)):
            with self.subTest(bound=bound),self.assertRaises(ValidationError):make_plan(bound,policy())

    def test_probe_strata_cover_candidates_once_without_overlap(self):
        config=policy();bound=binding(0,512);plan=make_plan(bound,config)
        candidates=tuple(range(128,512))
        self.assertEqual(len(plan.probe_positions),8)
        self.assertEqual(tuple(sorted(set(plan.probe_positions))),plan.probe_positions)
        for index,position in enumerate(plan.probe_positions):
            lo=index*len(candidates)//8;hi=(index+1)*len(candidates)//8
            self.assertIn(position,candidates[lo:hi])

    def test_probe_selection_stable_across_request_phase_and_data_independent(self):
        config=policy();bound=binding();first=make_plan(bound,config)
        changed=replace(bound,request_id='request-b',phase='different-phase',document_start=999,
                        request_sha256='f'*64)
        second=make_plan(changed,config)
        self.assertNotEqual(first.binding_sha256,second.binding_sha256)
        self.assertEqual(first.probe_positions,second.probe_positions)
        self.assertEqual(first,make_plan(bound,config))
        seeded=make_plan(bound,replace(config,seed='another-fixed-seed'))
        self.assertNotEqual(first.probe_positions,seeded.probe_positions)

    def test_probe_count_capped_by_available_nonmandatory_rows(self):
        config=policy(block_size=16,head_guard=0,tail_guard=0,probe_count=20)
        bound=binding(0,7,token_count=7)
        plan=make_plan(bound,config)
        self.assertEqual(plan.probe_positions,tuple(range(7)))

    def test_plan_constructor_rejects_invalid_positions(self):
        base=make_plan(binding(),policy())
        for change in ({'positions':[]},{'positions':()},
                       {'probe_positions':(600,600)},{'probe_positions':(600,599)},
                       {'probe_positions':(2000,)},{'probe_positions':(True,)},
                       {'mandatory_positions':base.probe_positions},{'binding_sha256':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValidationError):replace(base,**change)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.bound=binding();self.config=policy();self.plan=make_plan(self.bound,self.config)
        self.online=observation(self.bound,self.plan)
        self.cached=self.online.rows

    def check(self,cached=None,online=None,config=None,plan=None,bound=None):
        return evaluate(self.plan if plan is None else plan,self.config if config is None else config,
            self.cached if cached is None else cached,self.online if online is None else online,
            self.bound if bound is None else bound)

    def test_exact_probes_pass_only_explicit_policy(self):
        result=self.check(config=policy(relative_tolerance=0.,absolute_tolerance=0.))
        self.assertTrue(result.allow_reuse)
        self.assertEqual(result.reason,'within_probe_limits')
        self.assertEqual((result.max_relative_l2,result.max_absolute_error),(0.,0.))

    def test_worst_probe_not_average_controls_rejection(self):
        rows=list(self.cached);rows[-1]=(4.,6.)
        result=self.check(cached=rows)
        self.assertFalse(result.allow_reuse)
        self.assertAlmostEqual(result.max_relative_l2,5/math.sqrt(5))
        self.assertEqual(result.max_absolute_error,4.)

    def test_absolute_and_relative_requirements_are_both_enforced(self):
        cached=tuple((1.1,2.) for _ in self.plan.probe_positions)
        for config in (policy(relative_tolerance=1.,absolute_tolerance=.05),
                       policy(relative_tolerance=.01,absolute_tolerance=1.)):
            with self.subTest(config=config):self.assertFalse(self.check(cached=cached,config=config).allow_reuse)
        self.assertTrue(self.check(cached=cached,config=policy(relative_tolerance=.1,absolute_tolerance=.2)).allow_reuse)

    def test_threshold_equality_and_online_reference_denominator(self):
        online=observation(self.bound,self.plan,(3.,4.))
        cached=tuple((0.,0.) for _ in self.plan.probe_positions)
        result=self.check(cached=cached,online=online,config=policy(relative_tolerance=1.,absolute_tolerance=4.))
        self.assertTrue(result.allow_reuse)
        self.assertEqual(result.max_relative_l2,1.)
        self.assertEqual(result.max_absolute_error,4.)

    def test_explicit_norm_floor_for_zero_reference(self):
        online=observation(self.bound,self.plan,(0.,0.))
        cached=tuple((.01,0.) for _ in self.plan.probe_positions)
        result=self.check(cached=cached,online=online,config=policy(norm_floor=.1,
            relative_tolerance=.2,absolute_tolerance=.02))
        self.assertTrue(result.allow_reuse)
        self.assertAlmostEqual(result.max_relative_l2,.1)

    def test_changed_plan_or_binding_is_not_trusted(self):
        other=replace(self.bound,phase='foreign')
        with self.assertRaises(ValidationError):self.check(bound=other)
        changed_context=replace(self.bound,request_sha256='f'*64)
        with self.assertRaises(ValidationError):self.check(bound=changed_context)
        altered=replace(self.plan,probe_positions=tuple(range(512,520)))
        if altered==self.plan:altered=replace(self.plan,probe_positions=tuple(range(520,528)))
        with self.assertRaises(ValidationError):self.check(plan=altered)
        with self.assertRaises(ValidationError):self.check(config=policy(seed='tampered'))

    def test_online_binding_coverage_and_width_are_exact(self):
        malformed=(replace(self.online,binding_sha256='f'*64),
                   replace(self.online,positions=self.online.positions[:-1],rows=self.online.rows[:-1]),
                   replace(self.online,rows=tuple((1.,) for _ in self.plan.probe_positions)))
        for online in malformed:
            with self.subTest(online=online),self.assertRaises(ValidationError):self.check(online=online)

    def test_cached_count_shape_type_and_finiteness_are_strict(self):
        bad_values=(self.cached[:-1],tuple((1.,) for _ in self.cached),
                    tuple((float('nan'),2.) for _ in self.cached),
                    tuple((float('inf'),2.) for _ in self.cached),
                    tuple((True,2.) for _ in self.cached),
                    tuple((10**1000,2.) for _ in self.cached),'invalid')
        for cached in bad_values:
            with self.subTest(cached=cached),self.assertRaises(ValidationError):self.check(cached=cached)

    def test_finite_inputs_with_nonfinite_metric_deny_instead_of_passing(self):
        online=observation(self.bound,self.plan,(1e308,0.))
        cached=tuple((-1e308,0.) for _ in self.cached)
        result=self.check(cached=cached,online=online)
        self.assertFalse(result.allow_reuse)
        self.assertEqual(result.reason,'nonfinite_metric')
        self.assertIsNone(result.max_relative_l2)

    def test_reference_norm_overflow_is_fail_closed_even_for_equal_rows(self):
        online=observation(self.bound,self.plan,(1.3e308,1.3e308))
        result=self.check(cached=online.rows,online=online)
        self.assertFalse(result.allow_reuse)
        self.assertEqual(result.reason,'nonfinite_metric')
        self.assertIsNone(result.max_absolute_error)

    def test_ratio_overflow_cannot_pass_threshold_comparison(self):
        online=observation(self.bound,self.plan,(0.,0.))
        cached=tuple((1e308,0.) for _ in self.cached)
        result=self.check(cached=cached,online=online,config=policy(norm_floor=1e-308,
            relative_tolerance=1e308,absolute_tolerance=1e308))
        self.assertFalse(result.allow_reuse)
        self.assertEqual(result.reason,'nonfinite_metric')

    def test_probe_pass_explicitly_does_not_certify_unobserved_rows(self):
        result=self.check()
        self.assertTrue(result.allow_reuse)
        self.assertLess(len(self.plan.probe_positions)+len(self.plan.mandatory_positions),len(self.plan.positions))
        # Deliberately no all-row equality or model-accuracy certification field.
        self.assertFalse(hasattr(result,'all_rows_equivalent'))
        self.assertFalse(hasattr(result,'accuracy_certified'))

    def test_invalid_decision_metrics_cannot_claim_reuse(self):
        for args in ((True,'unobserved',None,None),(False,'bad',0.,None),
                     (True,'bad',float('nan'),0.),(1,'bad',0.,0.)):
            with self.subTest(args=args),self.assertRaises(ValidationError):GateDecision(*args)

    def test_clean_import_has_no_gpu_or_array_runtime_dependencies(self):
        code=('import sys; import kv_cache_engram.gate; '
              'assert not any(n.split(".")[0] in ("torch","numpy","triton","deepinfer") for n in sys.modules)')
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
