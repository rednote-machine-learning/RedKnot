from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GatedEngram'))
sys.path.insert(0, str(ROOT))
from kv_cache_engram.core import token_digest, digest
from kv_cache_engram.s1_runtime import S1Controller, s1_documents
from kv_cache_engram.s1_system_runtime import S1SystemController
from kv_cache_engram.s11_qualification import S11QualificationController
from kv_cache_engram.s1_cache import SWA_LAYERS, GLOBAL_SOURCES
from redknot.admission import plan_request as raw_plan_request, s2_documents, validate_plan, uniform_bank_budget, MAX_BANK_BYTES
from redknot.controller import adapted_controller_class, MemoryCaptureCache, MEMORY_ONLY_CAPTURE
from redknot.gate import verify_sparse_pair

SCOPE = dict(model_sha256='a'*64, execution_sha256='b'*64, tokenizer_sha256='c'*64)


def plan_request(ids, documents, scope, **kwargs):
    return raw_plan_request(ids, documents, scope, bank_budget_bytes=kwargs.pop('bank_budget_bytes', MAX_BANK_BYTES), **kwargs)


def document(ids, a, b):
    key = token_digest(ids[a:b])
    return dict(input_start=a, input_end=b, document_id=key, token_ids_sha256=key,
                offline_position=0, online_position=a, tokens=b-a)


class FakeCache:
    def __init__(self):
        self.bank = {}
        self.capture = None
        self.global_bases, self.global_carry, self.global_events = {'old': 1}, {'old': 1}, [1]
        self.preparation_s = 1
        self.phase_generation = 0

    def entry(self, key, kind, layer):
        value = self.bank[key, kind, layer]
        if not value['complete']:
            raise ValueError('incomplete entry')
        return value['tensor']

    def configure(self, *, enabled):
        self.enabled = enabled

    def begin_phase(self, phase, *, build_global_base):
        self.phase_generation += 1
        self.phase = phase
        self.build_global_base = build_global_base


def controller():
    layers = [NS(layer_idx=i) for i in range(40)]
    model = NS(layers=layers, yoco_boundary_layer_idx=3)
    c = adapted_controller_class()(layers, native_model=model, **SCOPE)
    c.state = 'COMPLETE'
    c.cache = FakeCache()
    c.attention = NS(configure=lambda **k: None, begin_phase=lambda: None)
    c.original_boundary = 3
    return c


def populate(c, key, length):
    for kind, layers in (('swa_halo', SWA_LAYERS), ('global_raw', GLOBAL_SOURCES)):
        for layer in layers:
            c.cache.bank[key, kind, layer] = dict(length=length, complete=True,
                tensor=NS(numel=lambda: 1, element_size=lambda: 2), version=0)


class AdmissionTests(unittest.TestCase):
    def test_natural_one_and_eleven_documents_need_no_padding(self):
        for n in (1, 3, 11):
            ids = list(range(300*n+17))
            docs = [document(ids, 300*i, 300*(i+1)) for i in range(n)]
            plan = plan_request(ids, docs, SCOPE)
            self.assertEqual(plan['route'], 's2')
            self.assertEqual(len(plan['effective_documents']), n)
            self.assertEqual(plan['padding_tokens_added'], 0)
            self.assertEqual(plan['input_ids_sha256'], token_digest(ids))

    def test_repeated_content_is_one_bank_key_with_two_occurrences(self):
        ids = list(range(300)) + [777] + list(range(300)) + [888]
        docs = [document(ids, 0, 300), document(ids, 301, 601)]
        plan = plan_request(ids, docs, SCOPE)
        self.assertEqual(len(plan['unique_document_ids']), 1)
        self.assertEqual(len(plan['effective_documents']), 2)
        self.assertEqual(plan['group_selected_rows'], [258])
        self.assertEqual(plan['unique_bank_payload_bytes'], 37*128*512*2 + 9216*300)
        self.assertEqual([d['start'] for d in s2_documents(ids, docs)], [0, 301])

    def test_short_blocks_remain_unmodified_full_rows(self):
        ids = list(range(515))
        docs = [document(ids, 0, 200), document(ids, 202, 502)]
        plan = plan_request(ids, docs, SCOPE)
        self.assertEqual(plan['route'], 's2')
        self.assertEqual(plan['group_selected_rows'], [343])
        self.assertEqual(len(plan['short_documents']), 1)
        self.assertEqual(plan['effective_documents'][0]['input_start'], 202)
        with self.assertRaises(ValueError):
            s2_documents(ids, docs)

    def test_all_short_empty_and_capacity_overflow_fallback(self):
        for docs in ([], [document(list(range(20)), 1, 19)]):
            plan = plan_request(list(range(20)), docs, SCOPE)
            self.assertEqual(plan['fallback_reason'], 'no_cacheable_document')
            self.assertEqual(plan['effective_documents'], [])
        ids = list(range(16384))
        plan = plan_request(ids, [document(ids, 0, 256)], SCOPE)
        self.assertEqual(plan['fallback_reason'], 'group17_selected_rows_exceed_8192')
        self.assertEqual(plan['group_selected_rows'], [16256])
        self.assertEqual(plan['effective_documents'], [])

    def test_overlap_hash_mutation_and_foreign_scope_rejected(self):
        ids = list(range(800));docs = [document(ids, 0, 300), document(ids, 299, 600)]
        with self.assertRaises(ValueError): plan_request(ids, docs, SCOPE)
        bad = document(ids, 0, 300);bad['token_ids_sha256'] = 'd'*64
        with self.assertRaises(ValueError): plan_request(ids, [bad], SCOPE)
        plan = plan_request(ids, [document(ids, 0, 800)], SCOPE)
        changed = ids[:];changed[1] = 9999
        with self.assertRaises(ValueError): validate_plan(plan, input_ids=changed)
        with self.assertRaises(ValueError): validate_plan(plan, scope=dict(SCOPE, tokenizer_sha256='d'*64))

    def test_all_rank_memory_budget_and_no_eviction_fallback(self):
        self.assertEqual(uniform_bank_budget([1000]*7+[500]), 100)
        self.assertEqual(uniform_bank_budget([100*1024**3]*8), MAX_BANK_BYTES)
        with self.assertRaises(ValueError): uniform_bank_budget([1000]*7)
        ids = list(range(300));docs = [document(ids, 0, 300)]
        self.assertEqual(raw_plan_request(ids, docs, SCOPE)['fallback_reason'], 'bank_budget_unavailable')
        small = plan_request(ids, docs, SCOPE, bank_budget_bytes=1024)
        self.assertEqual(small['fallback_reason'], 'bank_budget_exceeded')
        self.assertFalse(small['eviction_performed'])
        key = docs[0]['document_id']
        hit = plan_request(ids, docs, SCOPE, bank_budget_bytes=100, resident_bank_bytes=80, resident_document_ids=[key])
        self.assertEqual(hit['route'], 's2')
        self.assertEqual(hit['predicted_miss_payload_bytes'], 0)


class ControllerTests(unittest.TestCase):
    def test_private_hierarchy_preserves_method_bytecode_and_original_globals(self):
        cls = adapted_controller_class()
        for original, private in ((S1Controller, cls.private_base_class),
                (S1SystemController, cls.private_system_class),
                (S11QualificationController, cls.private_qualification_class)):
            for name in ('finish',):
                self.assertIs(getattr(private, name).__code__, getattr(original, name).__code__)
        for original, private in ((S1Controller, cls.private_base_class), (S1SystemController, cls.private_system_class)):
            self.assertEqual(private.reset.s2_admission_rewrite_count, 1)
            self.assertIs(private._policy_model_forward.__code__, original._policy_model_forward.__code__)
            self.assertIs(private._observe_layer.__code__, original._observe_layer.__code__)
        self.assertIs(cls.private_qualification_class.reset.__code__, S11QualificationController.reset.__code__)
        self.assertIs(S1Controller.reset.__globals__['s1_documents'], s1_documents)
        self.assertIsNot(cls.private_base_class.reset.__globals__, S1Controller.reset.__globals__)
        self.assertIs(cls.private_base_class.reset.__globals__['s1_documents'], s2_documents)

    def test_request_reset_preserves_bank_and_native_gate_but_no_prompt_proof(self):
        c = controller();ids = list(range(300));plan = plan_request(ids, [document(ids, 0, 300)], SCOPE)
        key = plan['unique_document_ids'][0];populate(c, key, 300);bank = dict(c.cache.bank)
        plan = plan_request(ids, [document(ids, 0, 300)], SCOPE, **c.resident_bank_state())
        c.qualified = True;c.sparse_reference = {'old': 1};c.sparse_qualified = True
        c.system_qualified = {1: True, 2: True, 4: True, 17: True}
        self.assertEqual(c.begin_request(plan, request_id='one'), [])
        self.assertEqual(c.cache.bank, bank)
        self.assertTrue(c.qualified)
        self.assertFalse(c.sparse_reference)
        self.assertFalse(c.sparse_qualified)
        self.assertFalse(any(c.system_qualified.values()))
        self.assertEqual(c.cache.global_bases, {})
        self.assertEqual(c.expected_document_count, 1)
        with self.assertRaises(ValueError): c.begin_request(plan, request_id='one')

    def test_misses_deduplicate_and_partial_entries_are_not_hits(self):
        c = controller();ids = list(range(300))*2
        plan = plan_request(ids, [document(ids, 0, 300), document(ids, 300, 600)], SCOPE)
        key = plan['unique_document_ids'][0]
        self.assertEqual(c.begin_request(plan, request_id='one'), [key])
        populate(c, key, 300)
        self.assertEqual(c.cache_misses(plan), [])
        del c.cache.bank[key, 'global_raw', 8]
        with self.assertRaises(ValueError): c.cache_misses(plan)

    def test_active_request_cannot_reset_or_reuse(self):
        c = controller();ids = list(range(300));plan = plan_request(ids, [document(ids, 0, 300)], SCOPE)
        c.state = 'RUNNING'
        with self.assertRaises(ValueError): c.begin_request(plan, request_id='one')

    def test_full_fallback_uses_original_native_reset_without_ten_fake_docs(self):
        c = controller();ids = [1, 2, 3];plan = plan_request(ids, [], SCOPE)
        c.begin_request(plan, request_id='one')
        c.configure(system_enabled=False, batch_group_chunks=1)
        c.reset('one', 'native', arm='recomputed', input_ids=ids, documents=[], phase='full_fallback')
        self.assertEqual(c.mode, 'native')
        self.assertEqual(c.docs, ())
        self.assertFalse(c.system_enabled)
        self.assertIsNone(c.batch_schedule)
        self.assertFalse(c.cache.build_global_base)

    def test_memory_only_capture_keeps_native_bank_construction(self):
        cache = MemoryCaptureCache.__new__(MemoryCaptureCache)
        cache.capture = None;cache.bank = {}
        cache.begin_capture('a'*64, 300, output_dir=MEMORY_ONLY_CAPTURE)
        self.assertEqual(cache.capture['length'], 300)
        self.assertIsNone(cache.capture['output_dir'])

    def test_session_admission_does_not_forge_old_numeric_flags(self):
        # Deliberately synthetic scope fixture tests the admission branch only;
        # it is not passed through set_session_certificate or issued as a proof.
        certificate = dict(sha256='d'*64, bank_budget_bytes=MAX_BANK_BYTES,
            supported_geometry=dict(max_input_tokens=1000, max_document_occurrences=2,
                max_document_tokens=600, max_group_selected_rows=600))
        c = controller();ids = list(range(300))
        plan = plan_request(ids, [document(ids, 0, 300)], SCOPE, session_certificate=certificate)
        c.begin_request(plan, request_id='one')
        c.session_certificate = certificate
        c._s2_admission_context = ('sensitive', 'one', token_digest(ids), False)
        c.require_sparse_admission(False);c.require_system_admission(17)
        self.assertFalse(c.sparse_qualified)
        self.assertFalse(any(c.system_qualified.values()))
        self.assertTrue(c.session_admission_used)
        c._s2_admission_context = ('sensitive', 'foreign', token_digest(ids), False)
        with self.assertRaises(ValueError): c.require_sparse_admission(False)

    def test_session_out_of_domain_is_explicit_full_fallback_plan(self):
        certificate = dict(sha256='d'*64, supported_geometry=dict(max_input_tokens=1000,
            max_document_occurrences=1, max_document_tokens=300, max_group_selected_rows=128))
        ids = list(range(600));docs = [document(ids, 0, 300), document(ids, 300, 600)]
        plan = plan_request(ids, docs, SCOPE, session_certificate=certificate)
        self.assertEqual(plan['fallback_reason'], 'session_max_document_occurrences_exceeded')
        self.assertEqual(plan['effective_documents'], [])
        validate_plan(plan, input_ids=ids)


class NumericalGateTests(unittest.TestCase):
    def fixture(self):
        ids = list(range(300))
        plan = plan_request(ids, [document(ids, 0, 300)], SCOPE)
        fp = lambda shape: dict(shape=shape, dtype='bfloat16', bytes=2*shape[-1], sha256='e'*64)
        values = [dict(key=[0, 300, layer], value=dict(output=fp([300 if layer < 3 else 128, 16]),
                    lazy={name: fp([1, 300 if layer < 3 else 128, 16])
                          for name in ('residual', 'post_layer_mix', 'comb_res_mix', 'pre_layer_mix')}))
                  for layer in range(40)]
        source = dict(schema='s1-source2-cache-fingerprint-v1', rows=150, main_sha256='f'*64, index_k_sha256='a'*64)
        ref = dict(schema='s1-controller-v1', mode='reference_sparse', system_enabled=True, batch_group_chunks=1,
            input_ids_sha256=plan['input_ids_sha256'], full_input_tokens=300, s2_request_plan_sha256=plan['sha256'],
            cache_scope=SCOPE, first_three_full=True, decode_native=True, source2_native_cache_preserved=True,
            mathematical_policy_changed=False, sparse_identity_fingerprints=values,
            sparse_identity_check_count=40, sparse_identity_fingerprints_sha256=digest(values),
            source2_identity_events=[dict(chunk_start=0, chunk_end=300, before=source, after=copy.deepcopy(source), exact=True)],
            sparse_identity_exact=False, s2_request_id='one', s2_request_generation=1,
            bank_generation=0, offline_bank_scope_sha256='b'*64, phase_generation=2)
        candidate = copy.deepcopy(ref)
        candidate.update(mode='sensitive', batch_group_chunks=17, sparse_identity_exact=True,
                         system_numerically_qualified=True, phase_generation=3)
        tail = dict(schema='s2-dflash-tail-runtime-v1', enabled=True, integrated_all_rank_runtime=True,
            input_ids_sha256=plan['input_ids_sha256'], input_tokens=300, window_size=128,
            retained_positions=list(range(172,300)), native_main_cache_update_calls=1,
            selected_positions_unchanged=True, sampler_placeholder_features_used=False,
            fingerprint_before_decode=True, same_policy_reference_exact=False,
            target_layer_ids=[2,7], native_dflash_layer_ids=[40],
            coverage_by_target={'2':128,'7':128}, state_id=1,
            qualification_fingerprints=dict(target_features=fp([128,32]),
                valid_main_swa=[dict(layer=40,value=fp([128,16]))]))
        ref['dflash_tail'] = tail
        candidate['dflash_tail'] = dict(copy.deepcopy(tail),same_policy_reference_exact=True)
        record = dict(output_ids=[2]*32, output_ids_sha256=token_digest([2]*32), text='x', finish_reason='length')
        return plan, ids, ref, candidate, record

    def test_full_40_layer_four_mhc_and_generation_comparison(self):
        plan, ids, ref, candidate, record = self.fixture()
        out = verify_sparse_pair(plan, ids, ref, candidate, record, record)
        self.assertEqual(out['original_chunk_layer_checks'], 40)
        self.assertTrue(out['all_rank_service_admission_still_required'])
        self.assertFalse(out['full_model_equivalence_claim'])

    def test_changed_mhc_and_missing_member_rejected(self):
        for mutation in ('hash', 'missing'):
            plan, ids, ref, candidate, record = self.fixture()
            values = candidate['sparse_identity_fingerprints']
            if mutation == 'hash': values[7]['value']['lazy']['residual']['sha256'] = 'd'*64
            else: values.pop()
            candidate['sparse_identity_fingerprints_sha256'] = digest(values)
            with self.assertRaises(ValueError): verify_sparse_pair(plan, ids, ref, candidate, record, record)

    def test_source2_change_and_prior_prompt_proof_rejected(self):
        for mutation in ('source2', 'request'):
            plan, ids, ref, candidate, record = self.fixture()
            if mutation == 'source2': candidate['source2_identity_events'][0]['after']['main_sha256'] = 'd'*64
            else: candidate['s2_request_plan_sha256'] = 'd'*64
            with self.assertRaises(ValueError): verify_sparse_pair(plan, ids, ref, candidate, record, record)

    def test_early_eos_allowed_but_no_false_complete_length(self):
        plan, ids, ref, candidate, record = self.fixture()
        early = dict(output_ids=[2, 1], output_ids_sha256=token_digest([2, 1]), text='x', finish_reason='stop')
        self.assertEqual(verify_sparse_pair(plan, ids, ref, candidate, early, early)['generated_tokens'], 2)
        invalid = dict(early, finish_reason='length')
        with self.assertRaises(ValueError): verify_sparse_pair(plan, ids, ref, candidate, invalid, invalid)


if __name__ == '__main__': unittest.main()
