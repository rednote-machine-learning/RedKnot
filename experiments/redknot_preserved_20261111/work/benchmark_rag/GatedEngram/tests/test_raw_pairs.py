import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_pilot.raw_pairs import RawPairWriter


class RawPairsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.docs = [f"{i:064x}" for i in range(8)]

    def writer(self, **kwargs):
        result = RawPairWriter(Path(self.temp.name)/"raw", documents=self.docs,
                               layers=(3,), identity={"fixture": True}, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_partial_never_publishes(self):
        w = self.writer()
        w.append_bytes("capture_independent", 3, self.docs[0], 0, 1, bytes(1024))
        with self.assertRaises(RuntimeError): w.finish()
        self.assertFalse((w.directory/"pair.json").exists())

    def test_rejects_gap_then_poison(self):
        w = self.writer()
        with self.assertRaises(ValueError):
            w.append_bytes("capture_independent", 3, self.docs[0], 1, 1, bytes(1024))
        with self.assertRaises(RuntimeError):
            w.append_bytes("capture_independent", 3, self.docs[0], 0, 1, bytes(1024))

    def test_no_overwrite(self):
        self.writer()
        with self.assertRaises(FileExistsError): self.writer()

    def test_budget(self):
        w = self.writer(max_bytes=1024)
        w.append_bytes("capture_independent", 3, self.docs[0], 0, 1, bytes(1024))
        with self.assertRaises(ValueError):
            w.append_bytes("capture_independent", 3, self.docs[0], 1, 1, bytes(1024))

    def test_duplicate_row(self):
        w = self.writer()
        w.append_bytes("shadow_full", 3, self.docs[0], 0, 1, bytes(1024))
        with self.assertRaises(ValueError):
            w.append_bytes("shadow_full", 3, self.docs[0], 0, 1, bytes(1024))

    def test_unbound_document(self):
        w = self.writer()
        with self.assertRaises(ValueError):
            w.append_bytes("shadow_full", 3, "f"*64, 0, 1, bytes(1024))

    def test_shape_byte_count(self):
        w = self.writer()
        with self.assertRaises(ValueError):
            w.append_bytes("shadow_full", 3, self.docs[0], 0, 1, bytes(512))

    def test_closed(self):
        w = self.writer()
        w.close()
        with self.assertRaises(RuntimeError): w.finish()

    def test_bf16_nonfinite_bytes_poison(self):
        w = self.writer()
        with self.assertRaises(ValueError):
            w.append_bytes("shadow_full", 3, self.docs[0], 0, 1, b"\x80\x7f" + bytes(1022))
        with self.assertRaises(RuntimeError): w.finish()


if __name__ == "__main__": unittest.main()
