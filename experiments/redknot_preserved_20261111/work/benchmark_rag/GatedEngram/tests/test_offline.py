import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kv_cache_engram.core import ValidationError, canonical_bytes, digest
from kv_cache_engram.deepinfer_adapter import CPUReferenceCaptureOps
from kv_cache_engram.offline import (
    CAPTURE_POINT, CaptureSpec, Snapshot, SnapshotWriter, capture_snapshot,
)


def spec(**changes):
    fields = dict(request_id="test-request", phase="offline-cpu-fixture",
                  model_sha256="1" * 64, tokenizer_sha256="2" * 64,
                  execution_sha256="3" * 64, input_ids=(10, 11, 12, 13),
                  document_start=0, document_end=4, layers=(3, 4), width=2)
    fields.update(changes)
    return CaptureSpec(**fields)


def provenance(value, **changes):
    output = dict(
        adapter="deepinfer-v4-swa-capture-v1", capture_point=CAPTURE_POINT,
        request_id=value.request_id, phase=value.phase, input_ids_sha256=value.request_sha256,
        input_token_count=len(value.input_ids), document_start=value.document_start,
        document_end=value.document_end, operator="python-float64-rmsnorm-reference",
        native_operator=False, accelerator_capture=False, streaming=True,
        chunks=[dict(layer=l, start=0, end=len(value.input_ids),
                     document_rows=len(value.document_ids)) for l in value.layers],
        context_equivalence_certified=False, native_numerical_alignment_measured=False,
        identity_sha256_by_layer={str(k): v.sha256 for k, v in value.identities.items()})
    output.update(changes)
    return output


class FakeAttention:
    use_swa = True
    window_size = 128
    is_kv_source = False
    is_index_source = False
    is_index_k_source = False
    is_yoco_hidden_consumer_layer = False
    yoco_swa_share_hidden = False

    def __init__(self, layer, width):
        self.layer_idx = layer
        self.qk_head_dim = width
        self.kv_swa_norm = SimpleNamespace(weight=[1.0] * width, variance_epsilon=1e-5, out_scale=1.0)
        self.calls = 0
        self.output = object()

    def _prepare_swa_kv(self, raw, cache, metadata, use_cp):
        self.calls += 1
        return self.output


def fake_driver(value, session):
    # Actually exercise the native interception contract, not writer.append.
    for start, end in ((0, 2), (2, len(value.input_ids))):
        pivots, indices = [], []
        for local, position in enumerate(range(start, end)):
            abs_lo = max(position - 127, 0)
            start_slot = min(abs_lo, start) - (start - 128)
            k_lo = max(0, abs_lo - start)
            row = (start_slot, 128 - start_slot, 128 + k_lo, local - k_lo + 1)
            visible = tuple(range(row[0], row[0] + row[1])) + tuple(range(row[2], row[2] + row[3]))
            pivots.append(row)
            indices.append(visible + (-1,) * (128 - len(visible)))
        for layer, attention in session.attentions.items():
            metadata = SimpleNamespace(
                is_prompt=True, num_reqs=1, total_q=end - start,
                positions=list(range(start, end)), cu_seqlens_q=[0, end - start],
                total_k_swa=128, kv_swa_full_index_pivots=tuple(pivots),
                kv_swa_full_index=tuple(indices),
                canonical_request_binding=session.chunk_marker(start, end))
            raw = [[float(pos + 1), float(layer)] for pos in range(start, end)]
            result = attention._prepare_swa_kv(raw, None, metadata, False)
            assert result is attention.output


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def path(self, name="snapshot"):
        return self.root / name

    def build(self, value=None, name="snapshot"):
        value = value or spec()
        writer = SnapshotWriter(self.path(name), value)
        for layer in value.layers:
            rows = tuple((float(i), float(layer)) for i in range(len(value.document_ids)))
            writer.append(layer, 0, rows[:1])
            if len(rows) > 1:
                writer.append(layer, 1, rows[1:])
        result = writer.finalize(provenance(value))
        return value, result

    def mutate_manifest(self, fn):
        path = self.path() / "manifest.json"
        value = json.loads(path.read_bytes())
        fn(value)
        value["seal"] = digest({k: v for k, v in value.items() if k != "seal"})
        path.write_bytes(canonical_bytes(value))

    def test_spec_is_immutable_and_binds_input(self):
        ids = [10, 11, 12, 13]
        value = spec(input_ids=ids)
        ids[0] = 99
        self.assertEqual(value.input_ids, (10, 11, 12, 13))
        self.assertNotEqual(value.request_sha256, spec(input_ids=(9, 11, 12, 13)).request_sha256)
        self.assertEqual(set(value.identities), {3, 4})

    def test_independent_requires_complete_position_zero_document(self):
        for kwargs in (dict(document_start=1), dict(document_end=3)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                spec(**kwargs)

    def test_invalid_spec(self):
        for kwargs in (dict(layers=(2,)), dict(layers=(3, 3)), dict(layers=(4, 3)),
                       dict(layers=(True,)), dict(width=True), dict(input_ids=(True, 2, 3, 4)),
                       dict(kind="unknown"), dict(document_end=5), dict(phase="")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                spec(**kwargs)

    def test_streaming_roundtrip_exact_little_endian_payload(self):
        value, sealed = self.build()
        raw = (self.path() / "layer3.f64le").read_bytes()
        self.assertEqual(raw[:16], struct.pack("<2d", 0.0, 3.0))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sealed["files"]["3"]["sha256"])
        with Snapshot(self.path(), expected_spec=value) as snapshot:
            self.assertEqual(snapshot.read_rows(3, 1, 3), ((1.0, 3.0), (2.0, 3.0)))
            artifact = snapshot.as_cache(3)
            artifact.validate(value.identities[3])
            self.assertEqual(artifact.rows, tuple((float(i), 3.0) for i in range(4)))

    def test_full_context_reference_never_as_cache(self):
        value, _ = self.build(spec(kind="full_context_reference", document_start=1, document_end=3))
        with Snapshot(self.path(), expected_spec=value) as snapshot:
            self.assertEqual(snapshot.read_rows(3, 0, 2), ((0.0, 3.0), (1.0, 3.0)))
            with self.assertRaises(ValidationError):
                snapshot.as_cache(3)

    def test_exclusive_directory_no_overwrite(self):
        self.build()
        before = (self.path() / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            SnapshotWriter(self.path(), spec())
        self.assertEqual(before, (self.path() / "manifest.json").read_bytes())

    def test_budget_rejected_before_directory_created(self):
        with self.assertRaises(ValidationError):
            SnapshotWriter(self.path(), spec(), max_payload_bytes=1)
        self.assertFalse(self.path().exists())

    def test_loader_payload_budget(self):
        self.build()
        with self.assertRaises(ValidationError):
            Snapshot(self.path(), max_payload_bytes=1)

    def test_read_value_budget_and_bounds(self):
        self.build()
        with Snapshot(self.path()) as snapshot:
            for args in ((3, 0, 4, 1), (3, 0, 5, 100), (3, 1, 1, 100), (2, 0, 1, 100)):
                with self.subTest(args=args), self.assertRaises(ValidationError):
                    snapshot.read_rows(*args)
            with self.assertRaises(ValidationError):
                snapshot.as_cache(3, max_values=1)

    def test_wrong_expected_spec(self):
        self.build()
        for value in (spec(request_id="other"), spec(phase="other"),
                      spec(input_ids=(90, 11, 12, 13)), spec(kind="full_context_reference")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Snapshot(self.path(), expected_spec=value)

    def test_partial_and_aborted_not_readable(self):
        writer = SnapshotWriter(self.path(), spec())
        writer.append(3, 0, ((1.0, 2.0),))
        writer.abort()
        self.assertEqual(writer.state, "ABORTED")
        with self.assertRaises(FileNotFoundError):
            Snapshot(self.path())
        with self.assertRaises(ValidationError):
            writer.append(3, 1, ((1.0, 2.0),))

    def test_incomplete_payload_cannot_finalize(self):
        writer = SnapshotWriter(self.path(), spec())
        writer.append(3, 0, ((1.0, 2.0),))
        with self.assertRaises(ValidationError):
            writer.finalize(provenance(spec()))
        self.assertEqual(writer.state, "ABORTED")
        self.assertFalse((self.path() / "manifest.json").exists())

    def test_bad_append_aborts(self):
        cases = [(3, 1, ((1.0, 2.0),)), (8, 0, ((1.0, 2.0),)),
                 (3, 0, ((1.0,),)), (3, 0, ((float("nan"), 2.0),)),
                 (3, 0, ((1.0, 2.0),) * 5), (True, 0, ((1.0, 2.0),))]
        for index, args in enumerate(cases):
            with self.subTest(args=args):
                writer = SnapshotWriter(self.path(str(index)), spec())
                with self.assertRaises(ValidationError):
                    writer.append(*args)
                self.assertEqual(writer.state, "ABORTED")

    def test_duplicate_rows_abort(self):
        writer = SnapshotWriter(self.path(), spec())
        writer.append(3, 0, ((1.0, 2.0),))
        with self.assertRaises(ValidationError):
            writer.append(3, 0, ((1.0, 2.0),))
        self.assertEqual(writer.state, "ABORTED")

    def test_bad_provenance_rejected_before_publication(self):
        mutations = [lambda p: p.update(request_id="foreign"),
                     lambda p: p.update(phase="foreign"),
                     lambda p: p.update(operator="unknown"),
                     lambda p: p.update(native_operator=True),
                     lambda p: p.update(context_equivalence_certified=True),
                     lambda p: p.update(streaming=False),
                     lambda p: p["chunks"].pop(),
                     lambda p: p["chunks"][0].update(start=1),
                     lambda p: p["chunks"][0].update(document_rows=3),
                     lambda p: p["identity_sha256_by_layer"].update({"3": "0" * 64}),
                     lambda p: p.update(input_token_count=True),
                     lambda p: p.update(unexpected="field")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                writer = SnapshotWriter(self.path(str(index)), spec())
                p = provenance(spec())
                mutate(p)
                with self.assertRaises(ValidationError):
                    writer.finalize(p)
                self.assertEqual(writer.state, "ABORTED")

    def test_payload_corruption_truncation_and_append(self):
        for index, mode in enumerate(("change", "truncate", "append")):
            _, _ = self.build(name=str(index))
            path = self.path(str(index)) / "layer3.f64le"
            raw = bytearray(path.read_bytes())
            if mode == "change":
                raw[0] ^= 1
            elif mode == "truncate":
                raw = raw[:-1]
            else:
                raw.extend(b"x")
            path.write_bytes(raw)
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                Snapshot(self.path(str(index)))

    def test_payload_nonfinite_even_when_checksum_matches(self):
        self.build()
        path = self.path() / "layer3.f64le"
        raw = struct.pack("<d", float("inf")) + path.read_bytes()[8:]
        path.write_bytes(raw)
        self.mutate_manifest(lambda m: m["files"]["3"].update(sha256=hashlib.sha256(raw).hexdigest()))
        with self.assertRaises(ValidationError):
            Snapshot(self.path())

    def test_read_revalidates_after_initial_success(self):
        self.build()
        with Snapshot(self.path()) as snapshot:
            path = self.path() / "layer3.f64le"
            raw = path.read_bytes()
            path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.assertRaises(ValidationError):
                snapshot.read_rows(3, 1, 2)

    def test_manifest_seal_and_malformed_rejected(self):
        self.build()
        path = self.path() / "manifest.json"
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"seal":"', b'"seal":"0', 1))
        with self.assertRaises(ValidationError):
            Snapshot(self.path())
        path.write_bytes(b'{"schema":1,"schema":2}')
        with self.assertRaises(ValidationError):
            Snapshot(self.path())
        path.write_bytes(b'{')
        with self.assertRaises(ValidationError):
            Snapshot(self.path())

    def test_manifest_strict_schema(self):
        self.build()
        self.mutate_manifest(lambda m: m.update(extra=True))
        with self.assertRaises(ValidationError):
            Snapshot(self.path())

    def test_derived_identity_mismatch(self):
        self.build()
        self.mutate_manifest(lambda m: m["spec"]["identities"]["3"].update(layer=4))
        with self.assertRaises(ValidationError):
            Snapshot(self.path())

    def test_unsafe_payload_paths(self):
        for index, name in enumerate(("../layer3.f64le", "/tmp/layer3.f64le", "layer4.f64le")):
            self.build(name=str(index))
            path = self.path(str(index)) / "manifest.json"
            m = json.loads(path.read_bytes())
            m["files"]["3"]["path"] = name
            m["seal"] = digest({k: v for k, v in m.items() if k != "seal"})
            path.write_bytes(canonical_bytes(m))
            with self.subTest(name=name), self.assertRaises(ValidationError):
                Snapshot(self.path(str(index)))

    def test_payload_and_manifest_symlinks_rejected(self):
        for index, filename in enumerate(("layer3.f64le", "manifest.json")):
            self.build(name=str(index))
            path = self.path(str(index)) / filename
            target = self.path(str(index)) / "saved"
            path.rename(target)
            path.symlink_to(target)
            with self.subTest(filename=filename), self.assertRaises(OSError):
                Snapshot(self.path(str(index)))

    def test_directory_symlink_rejected(self):
        self.build()
        alias = self.path("alias")
        alias.symlink_to(self.path(), target_is_directory=True)
        with self.assertRaises(OSError):
            Snapshot(alias)

    def test_nonregular_files_rejected_without_waiting_for_fifo_writer(self):
        for index, name in enumerate(("manifest.json", "layer3.f64le")):
            self.build(name=str(index))
            path = self.path(str(index)) / name
            path.unlink()
            os.mkfifo(path)
            with self.subTest(name=name), self.assertRaises(ValidationError):
                Snapshot(self.path(str(index)))

    def test_fsync_failure_never_publishes(self):
        writer = SnapshotWriter(self.path(), spec())
        for layer in spec().layers:
            writer.append(layer, 0, ((1.0, 2.0),) * 4)
        with mock.patch("kv_cache_engram.offline.os.fsync", side_effect=OSError("disk failed")):
            with self.assertRaises(OSError):
                writer.finalize(provenance(spec()))
        self.assertEqual(writer.state, "ABORTED")
        self.assertFalse((self.path() / "manifest.json").exists())

    def test_cpu_fake_driver_sink_error_restores_and_does_not_publish(self):
        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with mock.patch("kv_cache_engram.offline.SnapshotWriter.append", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                capture_snapshot(self.path(), value, attentions, fake_driver)
        self.assertFalse((self.path() / "manifest.json").exists())
        self.assertTrue(all("_prepare_swa_kv" not in vars(a) for a in attentions.values()))

    def test_completion_barrier_failure_does_not_publish(self):
        class FailingCompletion(CPUReferenceCaptureOps):
            def finish(self):
                raise RuntimeError("asynchronous prefill completion failed")

        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with self.assertRaisesRegex(RuntimeError, "completion failed"):
            capture_snapshot(self.path(), value, attentions, fake_driver, ops=FailingCompletion())
        self.assertTrue(all(a.calls == 2 for a in attentions.values()))
        self.assertTrue(all("_prepare_swa_kv" not in vars(a) for a in attentions.values()))
        self.assertFalse((self.path() / "manifest.json").exists())
        with self.assertRaises(FileNotFoundError):
            Snapshot(self.path())

    def test_completion_barrier_precedes_manifest_publication(self):
        calls = []
        target = self.path()

        class SuccessfulCompletion(CPUReferenceCaptureOps):
            def finish(self):
                calls.append("finished")
                if (target / "manifest.json").exists():
                    raise AssertionError("snapshot published before full prefill completion")

        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with capture_snapshot(target, value, attentions, fake_driver, ops=SuccessfulCompletion()):
            self.assertEqual(calls, ["finished"])
            self.assertTrue((target / "manifest.json").exists())

    def test_dependency_fixture_provenance_is_not_native_capture(self):
        value = spec()
        writer = SnapshotWriter(self.path(), value)
        for layer in value.layers:
            writer.append(layer, 0, ((1.0, 2.0),) * 4)
        writer.finalize(provenance(value, operator="deepinfer-norm-dependency-fixture"))
        with Snapshot(self.path()) as snapshot:
            p = snapshot.manifest["provenance"]
            self.assertEqual(p["operator"], "deepinfer-norm-dependency-fixture")
            self.assertIs(p["native_operator"], False)
            self.assertIs(p["accelerator_capture"], False)

    def test_dependency_fixture_cannot_claim_native_gpu(self):
        writer = SnapshotWriter(self.path(), spec())
        with self.assertRaises(ValidationError):
            writer.finalize(provenance(spec(), operator="deepinfer-norm-dependency-fixture",
                                       native_operator=True, accelerator_capture=True))
        self.assertFalse((self.path() / "manifest.json").exists())

    def test_pending_manifest_alone_is_not_publication(self):
        self.build()
        (self.path() / "manifest.json").unlink()
        self.assertTrue((self.path() / ".manifest.pending").exists())
        with self.assertRaises(FileNotFoundError):
            Snapshot(self.path())

    def test_publication_failure_leaves_unreadable_partial(self):
        writer = SnapshotWriter(self.path(), spec())
        for layer in spec().layers:
            writer.append(layer, 0, ((1.0, 2.0),) * 4)
        with mock.patch("kv_cache_engram.offline.os.link", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                writer.finalize(provenance(spec()))
        self.assertEqual(writer.state, "ABORTED")
        with self.assertRaises(FileNotFoundError):
            Snapshot(self.path())

    def test_completed_writer_is_single_use(self):
        writer = SnapshotWriter(self.path(), spec())
        for layer in spec().layers:
            writer.append(layer, 0, ((1.0, 2.0),) * 4)
        writer.finalize(provenance(spec()))
        for action in (lambda: writer.finalize(provenance(spec())), writer.abort,
                       lambda: writer.append(3, 4, ((1.0, 2.0),))):
            with self.assertRaises(ValidationError):
                action()

    def test_manifest_property_is_detached(self):
        self.build()
        with Snapshot(self.path()) as snapshot:
            view = snapshot.manifest
            view["files"]["3"]["sha256"] = "0" * 64
            self.assertEqual(snapshot.read_rows(3, 0, 1), ((0.0, 3.0),))

    def test_real_adapter_cpu_fake_driver_end_to_end(self):
        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with capture_snapshot(self.path(), value, attentions, fake_driver) as snapshot:
            self.assertEqual(snapshot.spec, value)
            self.assertEqual(len(snapshot.read_rows(3, 0, 4)), 4)
            self.assertEqual(snapshot.manifest["provenance"]["operator"], "python-float64-rmsnorm-reference")
            self.assertFalse(snapshot.manifest["provenance"]["native_operator"])
        self.assertTrue(all(a.calls == 2 and "_prepare_swa_kv" not in vars(a) for a in attentions.values()))

    def test_full_context_cpu_capture_extracts_document_only(self):
        value = spec(kind="full_context_reference", document_start=1, document_end=3)
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with capture_snapshot(self.path(), value, attentions, fake_driver) as snapshot:
            self.assertEqual(len(snapshot.read_rows(3, 0, 2)), 2)
            self.assertEqual(snapshot.manifest["provenance"]["input_token_count"], 4)
            with self.assertRaises(ValidationError):
                snapshot.as_cache(3)

    def test_driver_failure_after_all_layers_never_publishes(self):
        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}

        def failing_driver(v, session):
            fake_driver(v, session)
            raise RuntimeError("later model stage failed")

        with self.assertRaises(RuntimeError):
            capture_snapshot(self.path(), value, attentions, failing_driver)
        self.assertFalse((self.path() / "manifest.json").exists())
        self.assertTrue(all("_prepare_swa_kv" not in vars(a) for a in attentions.values()))
        with self.assertRaises(FileNotFoundError):
            Snapshot(self.path())

    def test_incomplete_driver_never_publishes(self):
        value = spec()
        attentions = {layer: FakeAttention(layer, value.width) for layer in value.layers}
        with self.assertRaises(ValidationError):
            capture_snapshot(self.path(), value, attentions, lambda v, s: None)
        self.assertFalse((self.path() / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
