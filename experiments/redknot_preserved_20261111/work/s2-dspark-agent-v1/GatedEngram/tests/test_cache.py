"""CPU storage and identity tests; no model, native runtime or network."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from kv_cache_engram.cache import CacheArtifact, decode_cache, load_cache, save_cache
from kv_cache_engram.core import (Binding, CacheIdentity, OnlineRows, ValidationError,
                                  normalize_rows, token_digest, validate_online)


def identity(**changes):
    value = CacheIdentity("a" * 64, "b" * 64, "c" * 64, token_digest([10, 20, 30, 40]), 3, 4, 2)
    return replace(value, **changes)


def binding(**changes):
    value = Binding("req", "prefill", identity(), 32790, 0, 4, "e" * 64)
    return replace(value, **changes)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[1.0, -2.0], [3.0, 4.0], [0.0, 5.0], [-6.0, 7.0]]
        self.cache = CacheArtifact.build(identity(), self.rows)

    def test_round_trip_bytes_and_disk(self):
        self.assertEqual(decode_cache(self.cache.to_bytes()), self.cache)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            save_cache(path, self.cache)
            self.assertEqual(load_cache(path), self.cache)

    def test_save_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            save_cache(path, self.cache)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_cache(path, self.cache)
            self.assertEqual(path.read_bytes(), before)

    def test_copy_and_frozen_ownership(self):
        self.rows[0][0] = 900
        self.assertEqual(self.cache.rows[0][0], 1.0)
        with self.assertRaises(FrozenInstanceError):
            self.cache.checksum = "d" * 64
        with self.assertRaises(TypeError):
            self.cache.rows[0][0] = 900

    def test_take_exact_doc_local_rows(self):
        self.assertEqual(self.cache.take((1, 3)), ((3.0, 4.0), (-6.0, 7.0)))
        self.assertEqual(self.cache.take(()), ())
        for positions in ((-1,), (4,), (True,), (1, 1), (3, 1)):
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                self.cache.take(positions)

    def test_reject_identity_mismatch(self):
        for field in ("model_sha256", "tokenizer_sha256", "execution_sha256", "tokens_sha256"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.cache.validate(identity(**{field: "f" * 64}))
        with self.assertRaises(ValidationError):
            self.cache.validate(identity(layer=4))

    def test_checksum_covers_rows_and_metadata(self):
        for change in ("rows", "identity", "checksum"):
            item = json.loads(self.cache.to_bytes())
            if change == "rows":
                item["rows"][0][0] = 1.1
            elif change == "identity":
                item["identity"]["layer"] = 4
            else:
                item["checksum"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValidationError):
                decode_cache(json.dumps(item).encode())

    def test_truncated_and_invalid_files(self):
        data = self.cache.to_bytes()
        for value in (data[:10], b"\xff", b"[]", b"null", data + b"{}", b""):
            with self.subTest(value=value[:12]), self.assertRaises(ValidationError):
                decode_cache(value)

    def test_duplicate_keys_are_rejected(self):
        data = self.cache.to_bytes().replace(b'"schema":', b'"schema":"x","schema":', 1)
        with self.assertRaises(ValidationError):
            decode_cache(data)

    def test_unknown_fields_and_schema_rejected(self):
        for kind in ("extra", "schema", "identity_extra", "identity_missing"):
            item = json.loads(self.cache.to_bytes())
            if kind == "extra":
                item["execute"] = "not executable"
            elif kind == "schema":
                item["schema"] = "unknown"
            elif kind == "identity_extra":
                item["identity"]["other"] = 1
            else:
                del item["identity"]["layer"]
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                decode_cache(json.dumps(item).encode())

    def test_bounded_load(self):
        data = self.cache.to_bytes()
        with self.assertRaises(ValidationError):
            decode_cache(data, max_bytes=len(data) - 1)
        self.assertEqual(decode_cache(data, max_bytes=len(data)), self.cache)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            save_cache(path, self.cache)
            with self.assertRaises(ValidationError):
                load_cache(path, max_bytes=10)

    def test_nonfinite_payload_rejected(self):
        for number in (float("nan"), float("inf"), float("-inf")):
            rows = [list(r) for r in self.rows]
            rows[0][0] = number
            with self.subTest(number=number), self.assertRaises(ValidationError):
                CacheArtifact.build(identity(), rows)
            item = json.loads(self.cache.to_bytes())
            item["rows"][0][0] = number
            with self.assertRaises(ValidationError):
                decode_cache(json.dumps(item).encode())

    def test_wrong_shape_and_scalar_type(self):
        for rows in ([], self.rows[:-1], [[1]] * 4, [[1, 2, 3]] * 4,
                     [[True, 2]] * 4, [["1", 2]] * 4, [[10 ** 1000, 0]] * 4):
            with self.subTest(rows=str(rows)[:30]), self.assertRaises(ValidationError):
                CacheArtifact.build(identity(), rows)

    def test_normalize_never_accepts_tensor_like_objects(self):
        class Dangerous:
            def __iter__(self):
                raise AssertionError("must not evaluate arbitrary device iterator")
        with self.assertRaises(ValidationError):
            normalize_rows(Dangerous())

    def test_direct_artifact_must_still_validate_checksum(self):
        forged = replace(self.cache, checksum="0" * 64)
        with self.assertRaises(ValidationError):
            forged.validate()
        with self.assertRaises(ValidationError):
            forged.to_bytes()


class ContractTests(unittest.TestCase):
    def test_hash_and_identity_validation(self):
        for value in ("", "A" * 64, "g" * 64, 12, None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                identity(model_sha256=value)
        for changes in ({"layer": True}, {"layer": -1}, {"width": 0}, {"token_count": 0},
                        {"representation": "post-rope"}, {"storage_dtype": "bf16"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                identity(**changes)

    def test_tokens_are_not_normalized_or_order_invariant(self):
        self.assertEqual(token_digest([10, 20]), token_digest((10, 20)))
        self.assertNotEqual(token_digest([10, 20]), token_digest([20, 10]))
        for ids in ([], [True], [-1], [1.0]):
            with self.subTest(ids=ids), self.assertRaises(ValidationError):
                token_digest(ids)

    def test_binding_covers_every_request_dimension(self):
        b = binding()
        for changes in ({"request_id": "other"}, {"phase": "quality"}, {"document_start": 0},
                        {"block_end": 3}, {"request_sha256": "f" * 64}, {"identity": identity(layer=4)}):
            with self.subTest(changes=changes):
                self.assertNotEqual(b.sha256, binding(**changes).sha256)
        self.assertEqual(b.positions, (0, 1, 2, 3))

    def test_binding_bounds_and_empty_request(self):
        for changes in ({"request_id": " "}, {"phase": ""}, {"document_start": -1},
                        {"block_start": -1}, {"block_start": 4}, {"block_end": 5},
                        {"block_end": 0}, {"request_sha256": "invalid"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                binding(**changes)

    def test_online_exact_binding_and_coverage(self):
        b = binding()
        good = OnlineRows(b.sha256, [0, 2], [[1, 2], [3, 4]])
        self.assertIs(validate_online(b, good, (0, 2)), good)
        for target, positions in ((binding(phase="other"), (0, 2)), (b, (0, 1)),
                                  (b, (-1, 2)), (b, (0, 4))):
            with self.subTest(target=target, positions=positions), self.assertRaises(ValidationError):
                validate_online(target, good, positions)
        with self.assertRaises(ValidationError):
            validate_online(b, OnlineRows(b.sha256, (0,), ((1.0,),)), (0,))

    def test_online_immutable_and_finite(self):
        values = [[1, 2]]
        online = OnlineRows(binding().sha256, [0], values)
        values[0][0] = 42
        self.assertEqual(online.rows, ((1.0, 2.0),))
        for positions in ((1, 0), (0, 0), (False,), (-1,)):
            with self.subTest(positions=positions), self.assertRaises(ValidationError):
                OnlineRows(binding().sha256, positions, [[1, 2]] * len(positions))
        with self.assertRaises(ValidationError):
            OnlineRows(binding().sha256, (0,), ((float("nan"), 0),))


if __name__ == "__main__":
    unittest.main()
