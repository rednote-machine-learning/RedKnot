"""Bounded streaming BF16 evidence, separate from the float64 CPU cache format.

Used only in untimed calibration captures. Each file covers every row of one
actual 32K cache document at one layer. Partial captures are never published as
complete pairs. No pickle, implicit conversion, overwrite, or GPU import.
"""
from __future__ import annotations

import hashlib
from array import array
import json
from pathlib import Path
import sys


MODES = ("capture_independent", "shadow_full")


def seal(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return dict(value, sha256=hashlib.sha256(raw).hexdigest())


class RawPairWriter:
    def __init__(self, directory, *, documents, layers, identity, max_bytes=8 * 1024**3):
        self.directory = Path(directory)
        self.documents = tuple(documents)
        self.layers = tuple(layers)
        if (len(self.documents) != 8 or len(set(self.documents)) != 8
                or not self.layers or len(set(self.layers)) != len(self.layers)
                or any(type(n) is not int or n not in tuple(range(3, 8))+tuple(range(9, 14))+tuple(range(15, 19)) for n in self.layers)
                or any(not isinstance(s, str) or len(s) != 64
                       or any(c not in "0123456789abcdef" for c in s) for s in self.documents)
                or type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("eight unique SHA256 document identities and bounded layer budget required")
        self.max_bytes, self.identity, self.bytes = max_bytes, dict(identity), 0
        self._streams, self._hashes, self._rows = {}, {}, {}
        self._poisoned = self._closed = False
        self.directory.mkdir(parents=True, exist_ok=False)

    def append_bytes(self, mode, layer, document, start, rows, data):
        """CPU-testable exact serialized BF16 append contract."""
        if not isinstance(data, bytes) or len(data) % 2:
            self._poisoned = True
            raise ValueError("even BF16 byte payload required")
        words = array("H")
        words.frombytes(data)
        if sys.byteorder != "little":
            words.byteswap()
        if any(word & 0x7f80 == 0x7f80 for word in words):
            self._poisoned = True
            raise ValueError("nonfinite BF16 byte payload")
        self._append_validated(mode, layer, document, start, rows, data)

    def _append_validated(self, mode, layer, document, start, rows, data):
        """Internal entry: finiteness already checked by bytes or tensor path."""
        if self._closed or self._poisoned:
            raise RuntimeError("capture is closed or poisoned")
        key = (mode, layer, document)
        try:
            if (mode not in MODES or layer not in self.layers or document not in self.documents
                    or type(start) is not int or type(rows) is not int
                    or rows <= 0 or start < 0 or start + rows > 32768
                    or not isinstance(data, bytes) or len(data) != rows * 512 * 2
                    or start != self._rows.get(key, 0)
                    or self.bytes + len(data) > self.max_bytes):
                raise ValueError("invalid, discontinuous, duplicate or over-budget BF16 capture")
            if key not in self._streams:
                path = self.directory / f"{mode}.layer{layer}.{document}.bf16"
                self._streams[key] = path.open("xb")
                self._hashes[key] = hashlib.sha256()
            written = self._streams[key].write(data)
            if written != len(data):
                raise OSError("short BF16 write")
            self._hashes[key].update(data)
            self._rows[key] = start + rows
            self.bytes += written
        except BaseException:
            self._poisoned = True
            raise

    def __call__(self, mode, layer, document, start, tensor):
        """Synchronous untimed D2H copy; never used in formal TTFT phases."""
        import torch
        if (sys.byteorder != "little" or not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.bfloat16 or tensor.ndim != 2
                or tensor.shape[1] != 512):
            self._poisoned = True
            raise ValueError("native little-endian BF16[N,512] required, no implicit cast")
        cpu = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(cpu).all()):
            self._poisoned = True
            raise ValueError("nonfinite native KV capture")
        self._append_validated(mode, layer, document, start, cpu.shape[0],
                               cpu.view(torch.uint8).numpy().tobytes())

    def finish(self):
        if self._closed or self._poisoned:
            raise RuntimeError("cannot publish closed/poisoned capture")
        expected = {(m, l, d) for m in MODES for l in self.layers for d in self.documents}
        if set(self._rows) != expected or any(n != 32768 for n in self._rows.values()):
            self._poisoned = True
            raise RuntimeError("both complete 8-document/layer captures required")
        self.close()
        files = []
        for (mode, layer, doc), h in sorted(self._hashes.items()):
            files.append(dict(mode=mode, layer=layer, document_sha256=doc,
                path=f"{mode}.layer{layer}.{doc}.bf16", rows=32768, width=512,
                bytes=32768*512*2, sha256=h.hexdigest()))
        value = seal(dict(schema="kv-cache-engram-paired-native-bf16-v1", complete=True,
            identity=self.identity, dtype="bfloat16-little-endian", representation="postnorm-prerope",
            data_origin_declaration="independent-document-and-full-context-native-capture",
            runtime_provenance_requires_external_sealed_phase=True,
            all_rows=True, files=files, bytes=self.bytes,
            floating_point_format_is_not_legacy_float64_snapshot=True))
        with (self.directory / "pair.json").open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            out.write("\n")
        return value

    def close(self):
        for stream in self._streams.values():
            stream.close()
        self._closed = True
