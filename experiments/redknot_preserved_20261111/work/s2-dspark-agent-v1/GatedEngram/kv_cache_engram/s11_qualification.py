"""Dense numerical oracle for the enabled S1.1 implementation.

This adapter changes qualification admission only.  Native/core/system operator
implementations stay untouched.  A reference is evidence to compare against;
it never grants a candidate configuration numerical qualification by itself.
"""
from __future__ import annotations

from types import FunctionType

from .core import require, token_digest
from .s1_runtime import S1Controller
from .s1_system_runtime import S1SystemController


ADAPTER_SCHEMA = "s11-system-enabled-dense-reference-v1"
REFERENCE_BACKEND = "system_enabled_dense_group1"


class S11QualificationController(S1SystemController):
    def reset(self, *args, system_empty_group_qualification=False,
              source_full_input_ids_sha256=None, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get("mode")
        if mode != "reference_sparse":
            require(mode != "sensitive" or self._configured_system,
                    "S1.1 forbids the retired original S1 policy arm")
            return super().reset(*args,
                system_empty_group_qualification=system_empty_group_qualification,
                source_full_input_ids_sha256=source_full_input_ids_sha256, **kwargs)

        require(self._configured_system is True and self._configured_group == 1,
                "S1.1 dense reference requires enabled system group1")
        require(kwargs.get("verify_sparse", False) is False,
                "a dense reference cannot verify or qualify itself")
        require(not any(self.system_qualified.values()),
                "S1.1 dense reference requires fresh configuration qualifications")
        self.cache.configure(enabled=True)
        self.attention.configure(enabled=True)
        if system_empty_group_qualification:
            require("empty_group" in kwargs.get("phase", ""),
                    "empty-group geometry is diagnostic-only")
            require(self._empty_group_case ==
                    (token_digest(kwargs["input_ids"]), source_full_input_ids_sha256),
                    "empty-group case/source binding differs")

            def validator(ids, documents, **unused):
                return self._empty_group_documents(ids, documents, source_full_input_ids_sha256)

            original = S1Controller.reset
            isolated = FunctionType(original.__code__,
                dict(original.__globals__, s1_documents=validator), original.__name__,
                original.__defaults__, original.__closure__)
            isolated.__kwdefaults__ = original.__kwdefaults__
            isolated(self, *args, **kwargs)
        else:
            # The original dense reset/forward still gathers full projections,
            # preserves dense FFN/mHC layout and fingerprints selected outputs.
            # Its virtual cache/attention calls use the enabled system objects.
            S1Controller.reset(self, *args, **kwargs)
        self.system_enabled, self.batch_group_chunks = True, 1
        self.batch_schedule = self._batch = self._active_deep_batch = None
        self._batch_step_trace, self._source2_before_batch = [], []

    def finish(self):
        report = super().finish()
        if self.mode == "reference_sparse":
            require(report["system_enabled"] is True and report["batch_group_chunks"] == 1
                    and report["system_numerically_qualified"] is False
                    and report["sparse_identity_exact"] is False,
                    "dense reference was incorrectly promoted to a qualified candidate")
        report.update(s11_qualification_adapter_schema=ADAPTER_SCHEMA,
                      s11_reference_backend=REFERENCE_BACKEND,
                      s11_original_s1_benchmark_executed=False)
        return report
