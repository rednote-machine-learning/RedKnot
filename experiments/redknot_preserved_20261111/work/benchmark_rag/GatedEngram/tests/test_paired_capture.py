"""Pair orchestration CPU fixtures; never model quality or native GPU tests."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError, canonical_bytes, digest
from kv_cache_engram.deepinfer_adapter import CPUReferenceCaptureOps
from kv_cache_engram.offline import CaptureSpec, Snapshot
from kv_cache_engram.paired_capture import (
    CLEAN_STATE_CONTRACT, FreshRun, PairCaptureSpec, PairedCapture, capture_pair, load_pair,
)
from test_deepinfer_adapter import FixtureAttention, metadata


def pair_spec(**changes):
    common = dict(model_sha256="a" * 64, tokenizer_sha256="b" * 64,
                  execution_sha256="c" * 64, layers=(3, 4), width=2)
    offline = CaptureSpec("r-independent", "offline", input_ids=(1, 2, 3, 4),
                          document_start=0, document_end=4, **common)
    online = CaptureSpec("r-reference", "reference", input_ids=(11, 12, 1, 2, 3, 4, 20),
                         document_start=2, document_end=6, kind="full_context_reference", **common)
    fields = dict(pair_id="pair-1", source_document_id="source/doc-1", split="calibration",
                  independent=offline, reference=online)
    fields.update(changes)
    return PairCaptureSpec(**fields)


def descriptor(spec, modules=None, driver=None, ops=None, run_id=None, **changes):
    modules = {layer: FixtureAttention(layer) for layer in spec.layers} if modules is None else modules
    def full_driver(value, session):
        for start, end in ((0, 2), (2, len(value.input_ids))):
            rows = tuple((float(token + 1), float(1 + sum(value.input_ids[:start + j])))
                         for j, token in enumerate(value.input_ids[start:end]))
            for module in modules.values():
                module._prepare_swa_kv(rows, None, metadata(session, start, end), False)
    fields = dict(spec=spec, attentions=modules, full_prefill_driver=full_driver if driver is None else driver,
                  run_id=run_id or "run-" + spec.request_id, clean_state_contract=CLEAN_STATE_CONTRACT,
                  initial_kv_state_empty=True, state_reused=False, ops=ops)
    fields.update(changes)
    return FreshRun(**fields)


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "pair"
        self.spec = pair_spec()
        self.events = []

    @contextmanager
    def factory(self, spec):
        run = descriptor(spec)
        self.events.append(("enter", spec.kind))
        try:
            yield run
        finally:
            self.assertTrue(all("_prepare_swa_kv" not in vars(a) for a in run.attentions.values()))
            self.events.append(("exit", spec.kind))

    def build(self, factory=None):
        return capture_pair(self.directory, self.spec, self.factory if factory is None else factory)

    def mutate(self, mutation, reseal=True):
        path = self.directory / "pair.json"
        value = json.loads(path.read_bytes())
        mutation(value)
        if reseal:
            value["seal"] = digest({k: v for k, v in value.items() if k != "seal"})
        path.write_bytes(canonical_bytes(value))

    def test_success_is_sequential_cleaned_up_and_loadable(self):
        with self.build() as pair:
            self.assertEqual(pair.spec, self.spec)
            self.assertEqual(pair.spec.source_document_id, "source/doc-1")
            self.assertEqual(pair.spec.split, "calibration")
            self.assertNotEqual(pair.independent.seal, pair.reference.seal)
            self.assertNotEqual(pair.independent.read_rows(3, 0, 1), pair.reference.read_rows(3, 0, 1))
            self.assertFalse(pair.manifest["gate_authorized"])
            self.assertFalse(pair.manifest["context_equivalence_certified"])
            for side in ("independent", "reference"):
                self.assertEqual(pair.manifest["captures"][side]["operator"], "python-float64-rmsnorm-reference")
                self.assertFalse(pair.manifest["captures"][side]["accelerator_capture"])
        self.assertEqual(self.events, [("enter", "independent_document"), ("exit", "independent_document"),
                                      ("enter", "full_context_reference"), ("exit", "full_context_reference")])
        with load_pair(self.directory, expected_spec=self.spec) as pair:
            self.assertEqual(len(pair.reference.read_rows(4, 0, 4)), 4)

    def test_pair_contract_mismatches_rejected(self):
        reference = self.spec.reference
        changes = ({"request_id": self.spec.independent.request_id}, {"model_sha256": "e" * 64},
                   {"tokenizer_sha256": "e" * 64}, {"execution_sha256": "e" * 64},
                   {"layers": (3,)}, {"width": 3}, {"input_ids": (11, 12, 1, 2, 9, 4, 20)})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                replace(self.spec, reference=replace(reference, **change))
        for change in ({"source_document_id": ""}, {"pair_id": ""}, {"split": "training"},
                       {"independent": reference}, {"reference": self.spec.independent}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                replace(self.spec, **change)

    def test_clean_state_must_be_explicit_and_bound(self):
        for change in ({"clean_state_contract": "best-effort"}, {"initial_kv_state_empty": False},
                       {"state_reused": True}, {"initial_kv_state_empty": 1}, {"state_reused": 0}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                descriptor(self.spec.independent, **change)

    def test_budget_checked_before_directory_and_factory(self):
        with self.assertRaises(ValidationError):
            capture_pair(self.directory, self.spec, lambda s: self.fail("factory must not run"), max_payload_bytes=1)
        self.assertFalse(self.directory.exists())

    def test_manifest_spec_budget_precedes_factory(self):
        with patch("kv_cache_engram.paired_capture.MAX_PAIR_MANIFEST_BYTES", 1):
            with self.assertRaisesRegex(ValidationError, "manifest byte budget"):
                self.build(lambda s: self.fail("factory must not run"))
        self.assertFalse(self.directory.exists())

    def test_existing_directory_never_overwritten(self):
        self.directory.mkdir()
        marker = self.directory / "owned"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.events, [])

    def test_foreign_spec_and_wrong_factory_type_fail_without_pair(self):
        @contextmanager
        def foreign(spec): yield descriptor(replace(spec, request_id=spec.request_id + "wrong"))
        for index, factory in enumerate((lambda spec: None, foreign)):
            self.directory = Path(self.temp.name) / str(index)
            with self.assertRaises(ValidationError): self.build(factory)
            self.assertFalse((self.directory / "pair.json").exists())

    def test_duplicate_run_id_or_attention_objects_rejected(self):
        for index, mode in enumerate(("id", "attention")):
            self.directory = Path(self.temp.name) / str(index)
            shared = {layer: FixtureAttention(layer) for layer in self.spec.independent.layers}
            @contextmanager
            def factory(spec):
                yield descriptor(spec, run_id="shared" if mode == "id" else None,
                                 modules=shared if mode == "attention" else None)
            with self.assertRaises(ValidationError): self.build(factory)
            self.assertTrue((self.directory / "independent" / "manifest.json").exists())
            self.assertFalse((self.directory / "pair.json").exists())

    def test_context_manager_instance_cannot_be_reentered_for_second_side(self):
        class Manager:
            def __enter__(inner): return descriptor(self.spec.independent)
            def __exit__(inner, *unused): return False
        manager = Manager()
        with self.assertRaisesRegex(ValidationError, "context manager"):
            self.build(lambda spec: manager)
        self.assertFalse((self.directory / "pair.json").exists())

    def test_reference_driver_tail_failure_preserves_first_side_only(self):
        runs = []
        @contextmanager
        def factory(spec):
            run = descriptor(spec)
            runs.append(run)
            original = run.full_prefill_driver
            def driver(value, session):
                original(value, session)
                if spec.kind == "full_context_reference": raise RuntimeError("model tail failed")
            yield replace(run, full_prefill_driver=driver)
        with self.assertRaisesRegex(RuntimeError, "tail failed"): self.build(factory)
        with Snapshot(self.directory / "independent") as snapshot: self.assertEqual(snapshot.spec, self.spec.independent)
        self.assertTrue((self.directory / "reference").exists())
        self.assertFalse((self.directory / "reference" / "manifest.json").exists())
        self.assertFalse((self.directory / "pair.json").exists())
        self.assertTrue(all("_prepare_swa_kv" not in vars(a) for run in runs for a in run.attentions.values()))

    def test_cleanup_failure_does_not_publish_pair(self):
        @contextmanager
        def factory(spec):
            yield descriptor(spec)
            if spec.kind == "full_context_reference": raise RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"): self.build(factory)
        self.assertTrue((self.directory / "reference" / "manifest.json").exists())
        self.assertFalse((self.directory / "pair.json").exists())

    def test_suppressed_driver_failure_is_not_success(self):
        class Manager:
            def __enter__(inner): return descriptor(self.spec.independent, driver=lambda *a: (_ for _ in ()).throw(RuntimeError("bad")))
            def __exit__(inner, *unused): return True
        with self.assertRaisesRegex(ValidationError, "suppressed"):
            self.build(lambda spec: Manager())
        self.assertFalse((self.directory / "pair.json").exists())

    def test_async_completion_failure_keeps_pair_unpublished(self):
        class Ops(CPUReferenceCaptureOps):
            def finish(self): raise RuntimeError("async completion failed")
        @contextmanager
        def factory(spec): yield descriptor(spec, ops=Ops())
        with self.assertRaisesRegex(RuntimeError, "async"): self.build(factory)
        self.assertFalse((self.directory / "pair.json").exists())

    def test_mixed_operators_rejected(self):
        class FixtureOps(CPUReferenceCaptureOps): label = "deepinfer-norm-dependency-fixture"
        @contextmanager
        def factory(spec):
            yield descriptor(spec, ops=FixtureOps() if spec.kind == "full_context_reference" else None)
        with self.assertRaisesRegex(ValidationError, "operators differ"): self.build(factory)
        self.assertFalse((self.directory / "pair.json").exists())

    def test_publication_failure_does_not_make_pair_readable(self):
        real_link = os.link
        def fail_pair(source, target, **kwargs):
            if target == "pair.json": raise OSError("pair publication failed")
            return real_link(source, target, **kwargs)
        with patch("kv_cache_engram.paired_capture.os.link", side_effect=fail_pair):
            with self.assertRaises(OSError): self.build()
        self.assertTrue((self.directory / ".pair.pending").exists())
        with self.assertRaises(FileNotFoundError): load_pair(self.directory)

    def test_pair_and_snapshot_tampering_rejected(self):
        with self.build(): pass
        self.mutate(lambda m: m.update(gate_authorized=True))
        with self.assertRaises(ValidationError): load_pair(self.directory)
        self.mutate(lambda m: m.update(gate_authorized=False))
        self.mutate(lambda m: m["captures"]["reference"].update(path="../elsewhere"))
        with self.assertRaises(ValidationError): load_pair(self.directory)

    def test_changed_snapshot_payload_rejected(self):
        with self.build(): pass
        path = self.directory / "reference" / "layer3.f64le"
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValidationError): load_pair(self.directory)

    def test_missing_layer_and_forged_source_rejected_against_expected(self):
        with self.build(): pass
        self.mutate(lambda m: m["spec"].update(source_document_id="foreign-source"))
        with self.assertRaisesRegex(ValidationError, "expected"): load_pair(self.directory, expected_spec=self.spec)
        self.mutate(lambda m: m["spec"].update(source_document_id=self.spec.source_document_id))
        (self.directory / "reference" / "layer4.f64le").unlink()
        with self.assertRaises(FileNotFoundError): load_pair(self.directory)

    def test_manifest_and_child_symlinks_rejected(self):
        for index, filename in enumerate(("pair.json", "reference")):
            self.directory = Path(self.temp.name) / str(index)
            with self.build(): pass
            path, saved = self.directory / filename, self.directory / (filename + "-saved")
            path.rename(saved)
            path.symlink_to(saved, target_is_directory=filename == "reference")
            with self.assertRaises(OSError): load_pair(self.directory)

    def test_pair_root_symlink_rejected(self):
        with self.build(): pass
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(OSError): load_pair(alias)

    def test_manifest_fifo_rejected_without_blocking(self):
        with self.build(): pass
        path = self.directory / "pair.json"
        path.unlink(); os.mkfifo(path)
        with self.assertRaises(ValidationError): load_pair(self.directory)

    def test_duplicate_json_and_seal_rejected(self):
        with self.build(): pass
        path = self.directory / "pair.json"
        original = path.read_bytes()
        path.write_bytes(b'{"seal":"bad","seal":"bad"}')
        with self.assertRaises(ValidationError): load_pair(self.directory)
        path.write_bytes(original)
        self.mutate(lambda m: m.update(seal="0" * 64), reseal=False)
        with self.assertRaises(ValidationError): load_pair(self.directory)

    def test_manifest_is_detached_and_close_closes_sources(self):
        pair = self.build()
        snapshot = pair.independent
        changed = pair.manifest
        changed["spec"]["split"] = "evaluation"
        self.assertEqual(pair.spec.split, "calibration")
        pair.close()
        with self.assertRaises(ValidationError): snapshot.read_rows(3, 0, 1)
        with self.assertRaises(ValidationError): _ = pair.reference
        pair.close()

    def test_validate_rechecks_payload_and_in_memory_scope(self):
        with self.build() as pair:
            pair.validate()
            pair.spec = replace(pair.spec, source_document_id="another-doc")
            with self.assertRaisesRegex(ValidationError, "spec changed"): pair.validate()
            pair.spec = self.spec
            path = self.directory / "independent" / "layer3.f64le"
            data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
            with self.assertRaises(ValidationError): pair.validate()

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch CPU runtime not installed")
    def test_torch_CPU_pair_without_native_runtime(self):
        import torch
        from kv_cache_engram.deepinfer_adapter import TorchCPUCaptureOps
        @contextmanager
        def factory(spec):
            modules = {layer: FixtureAttention(layer) for layer in spec.layers}
            for module in modules.values(): module.kv_swa_norm.weight = torch.ones(2, device="cpu")
            def driver(value, session):
                n = len(value.input_ids)
                m = metadata(session, 0, n)
                m.positions = torch.arange(n, device="cpu")
                m.cu_seqlens_q = torch.tensor([0, n], device="cpu")
                m.kv_swa_full_index_pivots = torch.tensor(m.kv_swa_full_index_pivots, device="cpu")
                m.kv_swa_full_index = torch.tensor(m.kv_swa_full_index, device="cpu")
                for module in modules.values():
                    module._prepare_swa_kv(torch.ones((n, 2), device="cpu"), None, m, False)
            yield descriptor(spec, modules=modules, driver=driver, ops=TorchCPUCaptureOps())
        with self.build(factory) as pair:
            self.assertEqual(pair.manifest["captures"]["reference"]["operator"], "torch-cpu-float32-rmsnorm-reference")
            self.assertFalse(pair.manifest["captures"]["reference"]["accelerator_capture"])


if __name__ == "__main__":
    unittest.main()
