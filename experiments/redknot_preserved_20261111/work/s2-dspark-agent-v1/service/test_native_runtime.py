"""CPU protocol tests against frozen admission functions; no mocked GPU proof."""
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE));sys.path.insert(0, str(STAGE/'GatedEngram'))
from service.native_runtime import NativeRuntime, TokenRequest, seal, digest, reuse_coverage, token_digest
from service.config import sources, GATED_SHA, NATIVE_SHA, validate_config
from kv_cache_engram.native_reuse import document_plan


class Controller:
    batch_schedule = None
    def configure(self, **kwargs):
        self.config = kwargs
    def reset(self, rid, mode, **kwargs):
        self.mode, self.args = mode, kwargs
        if mode == 'capture_independent':
            # Call the actual frozen admission to reproduce the documents=[] bug.
            document_plan(kwargs['input_ids'], kwargs['documents'], independent=True, natural_boundary_inputs=True)


class RuntimeTests(unittest.TestCase):
    def runtime(self, arm='s2'):
        r = NativeRuntime({}, arm, Path('/unused'), 0)
        r.allow_sync = contextlib.nullcontext
        r.driver = NS(gather=lambda value:[value]+[None]*7, barrier=lambda:None)
        return r

    def test_independent_capture_passes_native_document_contract(self):
        r = self.runtime();r.controller = Controller()
        r.batch_worker = NS(configure=lambda schedule:dict(enabled=False))
        req = TokenRequest('cold', tuple(range(512)), 1, 's2', 0)
        r._configure(req, 'cold', 'capture_independent')
        self.assertIsNone(r.controller.args['documents'])
        self.assertEqual(r.controller.config, dict(system_enabled=False,batch_group_chunks=1))
        self.assertIsNone(r.instrumentation)

    def test_full_phase_does_not_install_or_call_controller(self):
        r = self.runtime('full')
        r._configure(TokenRequest('full', (0,4), 1, 'full', 0), 'full', None)
        self.assertIsNone(r.controller)
        self.assertIsNone(r.batch_worker)
        with self.assertRaisesRegex(ValueError, 'Full'):
            r._configure(TokenRequest('bad', (0,4), 1, 'full', 0), 'bad', 'sensitive')

    def test_all_ranks_wait_after_publication_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.runtime();r.out = Path(tmp)
            checkpoints = []
            r.driver.barrier = lambda:checkpoints.append(json.loads((r.out/'proof.json').read_text()))
            original = seal(dict(passed=True))
            ref = r.publish('proof', original)
            self.assertEqual(checkpoints, [original])
            self.assertEqual(ref['sha256'],original['sha256'])
            # Existing sealed evidence must not be sealed a second time.
            self.assertEqual(checkpoints[0]['sha256'], digest(dict(passed=True)))

    def test_actual_frozen_source_trees_and_runtime_readbacks_match(self):
        root = os.environ.get('S2_EXPERIMENT_ROOT', str(STAGE/'runtime-readonly' if (STAGE/'runtime-readonly').exists() else STAGE.parent))
        src = sources(STAGE, root)
        admission = json.loads((STAGE/'adapted-gated-sources.json').read_text())
        self.assertEqual(len(src['native']),403)
        self.assertEqual(admission['parent_gated_sha256'],GATED_SHA)
        self.assertEqual(src['gated'],admission['gated'])
        self.assertEqual(admission['sha256'],digest({k:v for k,v in admission.items() if k != 'sha256'}))
        self.assertEqual(digest(src['native']),NATIVE_SHA)
        self.assertEqual(len(src['runtime']),5)
        self.assertFalse('torch' in sys.modules or 'deepinfer' in sys.modules)

    def test_static_prefix_and_moved_document_hits_are_not_conflated(self):
        key = token_digest((7,8,9))
        source = dict(input_start=2,input_end=5,tokens=3,document_id=key,
                      origin=dict(source_kind='checkpoint_tool_template'))
        original = (0,4,7,8,9,10)
        cold, pending = reuse_coverage(original,[source],[source],[],{})
        self.assertEqual(cold['eligible_hit_document_tokens'],0)
        history = {key:{pending[0][1]}}
        warm, unused = reuse_coverage(original,[source],[source],[key],history)
        self.assertEqual(warm['prefix_compatible_hit_document_tokens'],3)
        self.assertEqual(warm['nonprefix_hit_document_tokens'],0)
        moved = dict(source,input_start=3,input_end=6,origin=dict(source_kind='tool_result'))
        other, unused = reuse_coverage((0,5,6,7,8,9,10),[moved],[moved],[key],history)
        self.assertEqual(other['nonprefix_hit_document_tokens'],3)
        self.assertEqual(other['prefix_compatible_hit_document_tokens'],0)
        self.assertEqual(other['by_source_kind']['tool_result']['nonprefix_hit_document_tokens'],3)

    def test_cpu_config_rejects_unsealed_or_changed_execution(self):
        with self.assertRaisesRegex(ValueError, 'seal'):
            validate_config(dict(schema='s2-online-service-config-v1'))


if __name__ == '__main__':unittest.main()
