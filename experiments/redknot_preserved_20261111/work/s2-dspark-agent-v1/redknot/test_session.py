"""CPU contract tests; synthetic receipts never assert actual GPU qualification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'GatedEngram'))
sys.path.insert(0, str(ROOT))
from kv_cache_engram.core import digest
from redknot.session import (_case_file, _short_fallback, _bind_qualified, read_reference,
    validate_sources, validate_installation_process, REQUIRED_SOURCES)


class SessionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, body):
        value = dict(body, sha256=digest(body))
        path = self.root/(name+'.json')
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=value['sha256'])

    def short(self):
        sha = 'a'*64
        plan = self.write('short.plan', dict(route='full_fallback', effective_documents=[], input_ids_sha256=sha))
        formal = self.write('short.formal', dict(request_id='short', batch_schedule=None,
            batch_worker_reports=[dict(schema='s1-system-worker-execution-v1', rank=r, enabled=False, steps=[]) for r in range(8)],
            idle_reports=[None]*8, controller=dict(mode='native')))
        return dict(schema='s2-dspark-representative-actual-case-v1', case_id='short_only',
            actual_gpu_executed=True, official_task=False, qualified=None, input_ids_sha256=sha,
            record=dict(request_id='short', live_engine=True, replay=False, input_ids_sha256=sha,
                output_ids=[22,1], finish_reason='stop', preparation=dict(requested_arm='s2', effective_arm='full',
                    fallback_reason='no_cacheable_document', numerical_gate=None, unique_misses=0, captures=[], plan=plan),
                completion=dict(formal_evidence=formal, completed_all8_rank_admission=True)))

    def test_short_fallback_binds_native_disabled_all8_and_output(self):
        self.write('session-case.short_only', self.short())
        result = _short_fallback(self.root)
        self.assertFalse(result['complete_quality_claim'])
        self.assertEqual(set(result), {'case','plan','formal','complete_quality_claim'})
        self.assertEqual(len(result['formal']['file_sha256']), 64)

    def test_short_fallback_does_not_accept_failed_or_candidate_route(self):
        body = self.short()
        body['record']['preparation']['effective_arm'] = 's2'
        self.write('session-case.short_only', body)
        with self.assertRaises(ValueError): _short_fallback(self.root)
        body['record']['preparation']['effective_arm'] = 'full'
        body['record']['completion']['completed_all8_rank_admission'] = False
        self.write('session-case.short_only', body)
        with self.assertRaises(ValueError): _short_fallback(self.root)

    def test_reference_binds_exact_bytes_not_only_semantic_seal(self):
        ref = self.write('proof', dict(ok=True))
        _, bound = read_reference(ref)
        path = Path(ref['path']);path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError): read_reference(bound)

    def test_actual_case_cannot_be_relabelled_from_cpu_fixture(self):
        body = self.short();body['actual_gpu_executed'] = False
        self.write('session-case.short_only', body)
        with self.assertRaises(ValueError): _case_file(self.root, 'short_only')

    def test_native_tuple_member_chunks_use_sealed_json_equivalence(self):
        case = dict(case_id='natural_cold', plan=dict(input_ids_sha256='a'*64),
                    candidate_phase=dict(member_chunks=[(0,8192),(8192,9000)]))
        actual = dict(input_ids_sha256='a'*64,
                      qualified=json.loads(json.dumps({k:v for k,v in case.items() if k != 'case_id'})))
        _bind_qualified(actual, case)
        actual['qualified']['candidate_phase']['member_chunks'][1][1] = 9001
        with self.assertRaises(ValueError): _bind_qualified(actual, case)

    def test_complete_source_pins_and_live_controller_owner_required(self):
        sources = dict.fromkeys(REQUIRED_SOURCES, 'a'*64)
        validate_sources(sources)
        del sources['profile_group_contract.py']
        with self.assertRaises(ValueError): validate_sources(sources)
        owner = self.write('owner', dict(rank_processes=[dict(pid=-1, start_ticks=1, argv=['old'])]))
        with self.assertRaises(ValueError): validate_installation_process(dict(startup_evidence=dict(startup_session=owner)))


if __name__ == '__main__':
    unittest.main()
