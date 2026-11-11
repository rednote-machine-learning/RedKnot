"""Independent reader boundary tests; no model or GPU imports."""
import json
from pathlib import Path
import tempfile
import unittest

from gpu_pilot import read_tail_link_results as reader
from kv_cache_engram.core import digest


class EvidenceBoundaryTests(unittest.TestCase):
    def test_duplicate_json_key_rejected_before_trusting_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"raw.json"
            path.write_text('{"status":"complete","status":"failed","sha256":"'+'a'*64+'"}')
            with self.assertRaises(ValueError): reader.read(path,{})

    def test_valid_seal_does_not_accept_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            target, link = Path(folder)/"raw.json", Path(folder)/"link.json"
            body={"number":1};target.write_text(json.dumps(dict(body,sha256=digest(body))))
            link.symlink_to(target)
            with self.assertRaises(ValueError): reader.read(link,{})

    def test_read_records_raw_bytes_and_canonical_seals(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"raw.json";body={"number":1}
            value=dict(body,sha256=digest(body));path.write_text(json.dumps(value))
            evidence={}
            self.assertEqual(reader.read(path,evidence),value)
            self.assertEqual(evidence[path.name]["seal_sha256"],value["sha256"])
            self.assertEqual(len(evidence[path.name]["file_sha256"]),64)

    def test_negative_nonfinite_and_boolean_timing_are_not_numbers(self):
        for value in (True,False,-1,float("nan"),float("inf")):
            with self.subTest(value=value),self.assertRaises(ValueError):
                reader.number(value,"TTFT",True)

    def test_tail_mask_selects_no_middle_document_rows(self):
        positions=[0,1,32500,32641,32768,32769]
        selected,non_document,tail=reader.mask_reference(positions,0,32770,[(1,32769,"a"*64)])
        self.assertEqual(selected,[0,3,4,5])
        self.assertEqual(non_document,[0,5]);self.assertEqual(tail,[3,4])

    def test_raw_reference_cannot_escape_its_filename(self):
        body={"value":1};value=dict(body,sha256=digest(body))
        evidence={"raw.json":dict(file_sha256="a"*64,seal_sha256=value["sha256"])}
        reference=dict(file="../raw.json",file_sha256="a"*64,seal_sha256=value["sha256"])
        with self.assertRaises(ValueError):reader.verify_reference(reference,Path("raw.json"),value,evidence)

    def test_gold_scoring_does_not_turn_double_zero_into_success(self):
        self.assertEqual(reader.score("wrong answer",["Paris"]),dict(answer_em=0.,answer_token_f1=0.))
        self.assertEqual(reader.score("The Paris!",["Paris"]),dict(answer_em=1.,answer_token_f1=1.))


if __name__=="__main__":unittest.main()
