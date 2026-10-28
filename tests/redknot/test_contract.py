import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.redknot.contract import SCHEMA, digest, read_manifest, seal, tokenizer_identity, validate_manifest, write_json


def case(sample_id: str = "case", ids: list[int] | None = None) -> dict:
    tokens = ids or [10, 20, 30]
    answers = ["Miller v. California"]
    return {
        "sample_id": sample_id,
        "dataset": "hotpotqa",
        "input_ids": tokens,
        "input_ids_sha256": digest(tokens),
        "references": answers,
        "references_sha256": digest(answers),
        "max_new_tokens": 32,
        "scored": True,
        "ignore_eos": False,
    }


def manifest() -> dict:
    return seal(
        {
            "schema": SCHEMA,
            "model": {
                "profile": "deepseek-v4.1-flash",
                "model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "tokenizer": {"fingerprint": "frozen"},
            },
            "suite": {"name": "64k"},
            "warmup": [case("warmup", [40, 50])],
            "cases": [case()],
        }
    )


class ContractTests(unittest.TestCase):
    def test_seal_and_round_trip(self) -> None:
        value = manifest()
        self.assertEqual(validate_manifest(value), value)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_json(Path(temporary) / "manifest.json", value)
            self.assertEqual(read_manifest(path), value)
            with self.assertRaises(FileExistsError):
                write_json(path, value)

    def test_detects_modified_token_even_if_outer_seal_recomputed(self) -> None:
        value = manifest()
        value["cases"][0]["input_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "input token hash"):
            validate_manifest(seal(value))

    def test_reference_tampering(self) -> None:
        value = manifest()
        value["cases"][0]["references"] = ["different answer"]
        with self.assertRaisesRegex(ValueError, "reference hash"):
            validate_manifest(seal(value))

    def test_warmup_and_measured_are_disjoint(self) -> None:
        value = manifest()
        value["warmup"] = [case("different-id")]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_manifest(seal(value))

    def test_duplicate_case_identity_is_rejected(self) -> None:
        value = manifest()
        value["cases"].append(case())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(seal(value))

    def test_invalid_tokens_are_rejected_before_gpu(self) -> None:
        for tokens in ([True, 1], [-1, 1], [2**31], [1.5], []):
            with self.subTest(tokens=tokens):
                value = manifest()
                value["cases"][0].update(input_ids=tokens, input_ids_sha256=digest(tokens))
                with self.assertRaises(ValueError):
                    validate_manifest(seal(value))

    def test_showcase_cannot_enter_accuracy(self) -> None:
        value = manifest()
        value["cases"][0]["ignore_eos"] = True
        with self.assertRaisesRegex(ValueError, "showcase"):
            validate_manifest(seal(value))

    def test_source_seal_is_required(self) -> None:
        value = manifest()
        del value["sha256"]
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_manifest(value)

    def test_fingerprints_tokenizer_assets_not_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
            (root / "model.safetensors").write_bytes(b"not-a-weight-file")
            first = tokenizer_identity(root)
            (root / "model.safetensors").write_bytes(b"changed")
            self.assertEqual(first["fingerprint"], tokenizer_identity(root)["fingerprint"])
            (root / "tokenizer_config.json").write_text('{"bos_token": "<bos>"}')
            self.assertNotEqual(first["fingerprint"], tokenizer_identity(root)["fingerprint"])

    def test_fingerprints_separate_chat_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tokenizer.json").write_text("{}")
            initial = tokenizer_identity(root)["fingerprint"]
            (root / "chat_template.jinja").write_text("{{ messages }}")
            single = tokenizer_identity(root)["fingerprint"]
            self.assertNotEqual(initial, single)
            (root / "chat_templates").mkdir()
            (root / "chat_templates" / "tool_use.jinja").write_text("{{ tools }}")
            self.assertNotEqual(single, tokenizer_identity(root)["fingerprint"])


if __name__ == "__main__":
    unittest.main()
