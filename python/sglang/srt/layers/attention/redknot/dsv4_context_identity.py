from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple


CONTEXT_SEGMENT_SCHEMA = "redknot_pure_mla_context_segment_v1"
SHA256_PREFIX = "sha256:"
NATIVE_FULL_SCOPE_POLICY = "native_dsv4_full_candidate_scope_v1"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(SHA256_PREFIX):
        return False
    digest = value[len(SHA256_PREFIX) :]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return str(value)


def _require_hex64(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase 64-character hex digest")
    return value


def _token_bytes(token_ids: Iterable[int]) -> bytes:
    payload = bytearray()
    for raw_token_id in token_ids:
        if type(raw_token_id) is not int:
            raise TypeError("token ids must be built-in integers")
        token_id = raw_token_id
        if token_id < 0 or token_id >= 1 << 32:
            raise ValueError("token ids must fit in an unsigned 32-bit word")
        payload.extend(token_id.to_bytes(4, "little", signed=False))
    return bytes(payload)


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    return SHA256_PREFIX + hashlib.sha256(_token_bytes(token_ids)).hexdigest()


EMPTY_TOKEN_IDS_SHA256 = token_ids_sha256(())


def context_segment_sha256(
    *,
    execution_profile: str,
    head_scope_policy: str,
    model_compat_hash: str,
    head_policy_hash: str,
    token_hash: str,
    prefix_input_hash: str,
    full_input_hash: str,
    source_start: int,
    source_end: int,
    length: int,
    canonical_start_pos: int,
) -> str:
    """Derive the artifact key from both token content and causal context.

    ``token_hash`` remains the content identity consumed by the existing
    z_off/shared-latent stores.  ``seg_hash`` must additionally bind the exact
    prefix under which those rows were computed; equal 8K token bodies under
    different prefixes are deliberately different artifacts.
    """

    if not isinstance(execution_profile, str) or not execution_profile:
        raise ValueError("execution_profile must be non-empty")
    if head_scope_policy != NATIVE_FULL_SCOPE_POLICY:
        raise ValueError(
            "context-conditioned pure MLA requires native DSV4 full head scope"
        )
    model_compat_hash = _require_hex64(
        model_compat_hash, name="model_compat_hash"
    )
    head_policy_hash = _require_hex64(
        head_policy_hash, name="head_policy_hash"
    )
    token_hash = _require_sha256(token_hash, name="token_hash")
    prefix_input_hash = _require_sha256(
        prefix_input_hash, name="prefix_input_hash"
    )
    full_input_hash = _require_sha256(full_input_hash, name="full_input_hash")
    if any(isinstance(value, bool) or type(value) is not int for value in (
        source_start,
        source_end,
        length,
        canonical_start_pos,
    )):
        raise TypeError("context segment positions/length must be integers")
    if source_start < 0 or length <= 0 or source_end != source_start + length:
        raise ValueError("context segment must satisfy end=start+positive length")
    if canonical_start_pos != 0:
        raise ValueError("context-conditioned pure snapshots remain canonical at zero")
    canonical = {
        "canonical_start_pos": canonical_start_pos,
        "execution_profile": execution_profile,
        "head_scope_policy": head_scope_policy,
        "head_policy_hash": head_policy_hash,
        "full_input_hash": full_input_hash,
        "length": length,
        "model_compat_hash": model_compat_hash,
        "prefix_input_hash": prefix_input_hash,
        "schema": CONTEXT_SEGMENT_SCHEMA,
        "source_end": source_end,
        "source_start": source_start,
        "token_hash": token_hash,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContextSegmentContract:
    execution_profile: str
    head_scope_policy: str
    model_compat_hash: str
    head_policy_hash: str
    seg_hash: str
    token_hash: str
    prefix_input_hash: str
    full_input_hash: str
    source_start: int
    source_end: int
    length: int
    canonical_start_pos: int

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        execution_profile: str,
        head_scope_policy: str,
        model_compat_hash: str,
        head_policy_hash: str,
    ) -> "ContextSegmentContract":
        if not isinstance(raw, Mapping):
            raise TypeError("context segment metadata must be a mapping")
        try:
            contract = cls(
                execution_profile=str(execution_profile),
                head_scope_policy=str(head_scope_policy),
                model_compat_hash=str(model_compat_hash),
                head_policy_hash=str(head_policy_hash),
                seg_hash=str(raw["seg_hash"]),
                token_hash=str(raw["token_hash"]),
                prefix_input_hash=str(raw["prefix_input_hash"]),
                full_input_hash=str(raw["full_input_hash"]),
                source_start=raw["source_start"],
                source_end=raw["source_end"],
                length=raw["length"],
                canonical_start_pos=raw["canonical_start_pos"],
            )
        except KeyError as error:
            raise ValueError(
                f"context segment metadata is missing {error.args[0]}"
            ) from error
        contract.validate()
        return contract

    def validate(self) -> None:
        _require_sha256(self.seg_hash, name="seg_hash")
        _require_sha256(self.token_hash, name="token_hash")
        _require_sha256(self.prefix_input_hash, name="prefix_input_hash")
        _require_sha256(self.full_input_hash, name="full_input_hash")
        expected = context_segment_sha256(
            execution_profile=self.execution_profile,
            head_scope_policy=self.head_scope_policy,
            model_compat_hash=self.model_compat_hash,
            head_policy_hash=self.head_policy_hash,
            token_hash=self.token_hash,
            prefix_input_hash=self.prefix_input_hash,
            full_input_hash=self.full_input_hash,
            source_start=self.source_start,
            source_end=self.source_end,
            length=self.length,
            canonical_start_pos=self.canonical_start_pos,
        )
        if self.seg_hash != expected:
            raise ValueError("seg_hash is not bound to its prefix/full-input contract")


def validate_context_segment_chain(
    raw_segments: Sequence[Mapping[str, object]],
    *,
    execution_profile: str,
    head_scope_policy: str,
    model_compat_hash: str,
    head_policy_hash: str,
    offline_prefix_hash: Optional[str] = None,
) -> Tuple[ContextSegmentContract, ...]:
    if not isinstance(raw_segments, (tuple, list)) or not raw_segments:
        raise ValueError("context-conditioned restore needs ordered segments")
    contracts = tuple(
        ContextSegmentContract.from_mapping(
            raw,
            execution_profile=execution_profile,
            head_scope_policy=head_scope_policy,
            model_compat_hash=model_compat_hash,
            head_policy_hash=head_policy_hash,
        )
        for raw in raw_segments
    )
    cursor = 0
    expected_prefix_hash = EMPTY_TOKEN_IDS_SHA256
    for index, (raw, contract) in enumerate(zip(raw_segments, contracts)):
        if contract.source_start != cursor:
            raise ValueError(
                f"context segment {index} is not contiguous from zero: "
                f"expected {cursor}, got {contract.source_start}"
            )
        global_offset = raw.get("global_offset", contract.source_start)
        if type(global_offset) is not int or global_offset != contract.source_start:
            raise ValueError(
                f"context segment {index} global_offset differs from source_start"
            )
        if "skip_first" in raw and (
            type(raw["skip_first"]) is not int or raw["skip_first"] != 0
        ):
            raise ValueError(
                f"context segment {index} must restore every context-qualified row"
            )
        if contract.prefix_input_hash != expected_prefix_hash:
            raise ValueError(
                f"context segment {index} prefix hash does not chain from prior input"
            )
        cursor = contract.source_end
        expected_prefix_hash = contract.full_input_hash
    if offline_prefix_hash is not None:
        offline_prefix_hash = _require_sha256(
            offline_prefix_hash, name="offline_prefix_hash"
        )
        if offline_prefix_hash != expected_prefix_hash:
            raise ValueError("offline_prefix_hash differs from the final segment input")
    return contracts


def _canonical_plan_digest(plan: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("context plan is not canonically serializable") from error
    return SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _normalize_chunk(
    positions: Sequence[int], token_ids: Sequence[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    normalized_positions = tuple(positions)
    normalized_tokens = tuple(token_ids)
    if not normalized_positions or len(normalized_positions) != len(normalized_tokens):
        raise ValueError("context certificate needs one token for every non-empty row")
    if any(type(value) is not int for value in normalized_positions):
        raise TypeError("context certificate positions must be built-in integers")
    if any(type(value) is not int for value in normalized_tokens):
        raise TypeError("context certificate token ids must be built-in integers")
    if normalized_positions[0] < 0 or any(
        right != left + 1
        for left, right in zip(normalized_positions, normalized_positions[1:])
    ):
        raise ValueError("context certificate positions must be strictly contiguous")
    # Validate the integer/range contract now; the returned bytes are reused by
    # both the per-chunk and cumulative hash checks below.
    _token_bytes(normalized_tokens)
    return normalized_positions, normalized_tokens


@dataclass
class _ContextStreamState:
    mode: str
    request_id: str
    request_binding: Tuple[object, ...]
    plan_digest: str
    scheduler_total: int
    contracts: Tuple[ContextSegmentContract, ...]
    query_start: int
    request_input_hash: Optional[str]
    next_position: int = 0
    next_contract: int = 0
    cumulative_hasher: object = field(default_factory=hashlib.sha256, repr=False)
    input_verified: bool = False
    publication_receipt: Optional[str] = None
    poisoned: bool = False

    @property
    def cumulative_hash(self) -> str:
        return SHA256_PREFIX + self.cumulative_hasher.copy().hexdigest()

    def poison(self, message: str) -> None:
        self.poisoned = True
        raise ValueError(message)


@dataclass
class PreparedContextSnapshotPublication:
    """A fully validated receipt whose post-visibility commit cannot fail."""

    _state: _ContextStreamState = field(repr=False)
    receipt: str
    committed: bool = False

    def commit_noexcept(self) -> None:
        # All validation, JSON encoding and hashing happened before the
        # irreversible adapter.confirm boundary.  These two Python reference
        # assignments allocate no new semantic object and intentionally have
        # no branch that can raise.
        self._state.publication_receipt = self.receipt
        self.committed = True

    def poison_noexcept(self) -> None:
        """Retire a prepared receipt after an irreversible confirm failure."""

        self._state.poisoned = True


class ContextTokenStreamRegistry:
    """Fail-closed CPU certificate spanning SGLang prefill microforwards.

    The registry neither selects rows nor touches attention/KV state.  It binds
    the token stream to a scheduler-owned request lifecycle tuple, proves that
    snapshot target rows follow the declared causal prefix, and proves that a
    later restore consumes the identical frozen order/positions.  Input proof
    and snapshot publication proof are deliberately separate states.
    """

    def __init__(self, *, max_entries: int = 1024):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("context stream registry capacity must be positive")
        self._max_entries = max_entries
        self._states: "OrderedDict[Tuple[str, Tuple[object, ...]], _ContextStreamState]" = OrderedDict()

    def _install(
        self,
        key: Tuple[str, Tuple[object, ...]],
        state: _ContextStreamState,
    ) -> None:
        if len(self._states) >= self._max_entries:
            completed_key = next(
                (
                    candidate
                    for candidate, value in self._states.items()
                    if (
                        (value.mode == "restore" and value.input_verified)
                        or value.publication_receipt is not None
                        or value.poisoned
                    )
                ),
                None,
            )
            if completed_key is None:
                raise RuntimeError("context stream registry has no safely evictable entry")
            self._states.pop(completed_key)
        self._states[key] = state

    def _state(
        self,
        *,
        mode: str,
        request_id: str,
        request_binding: Tuple[object, ...],
        plan: Mapping[str, object],
        scheduler_total: int,
        contracts: Tuple[ContextSegmentContract, ...],
        query_start: int,
        request_input_hash: Optional[str],
        chunk_start: int,
        trusted_cached_prefix_tokens: Sequence[int] = (),
    ) -> _ContextStreamState:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("context certificate request_id must be non-empty")
        if (
            type(request_binding) is not tuple
            or len(request_binding) < 4
            or any(
                isinstance(value, (list, dict, set))
                for value in request_binding
            )
        ):
            raise ValueError(
                "context certificate needs an immutable scheduler request binding"
            )
        if type(scheduler_total) is not int or scheduler_total <= 0:
            raise ValueError("context certificate needs scheduler-owned total tokens")
        plan_digest = _canonical_plan_digest(plan)
        key = (mode, request_binding)
        state = self._states.get(key)
        if state is None:
            cached_prefix = tuple(trusted_cached_prefix_tokens)
            if chunk_start == 0:
                if cached_prefix:
                    raise ValueError(
                        "position-zero context stream has an unexpected cached prefix"
                    )
            elif len(cached_prefix) != chunk_start:
                raise ValueError(
                    "context token stream started after position zero without the "
                    "exact scheduler-owned cached prefix"
                )
            state = _ContextStreamState(
                mode=mode,
                request_id=request_id,
                request_binding=request_binding,
                plan_digest=plan_digest,
                scheduler_total=scheduler_total,
                contracts=contracts,
                query_start=query_start,
                request_input_hash=request_input_hash,
            )
            if cached_prefix:
                # ``cached_prefix`` is copied from ScheduleBatch.req.fill_ids by
                # ForwardBatch.init_new, after the radix matcher has established
                # the exact prefix length.  Rebuild the ordinary streaming SHA
                # state from those scheduler-owned ids instead of trusting a
                # client-provided digest or weakening the final request hash.
                cursor = 0
                while cursor < chunk_start:
                    if state.next_contract >= len(contracts):
                        state.poison(
                            "cached prefix extends beyond the offline contract chain"
                        )
                    contract = contracts[state.next_contract]
                    if (
                        contract.source_start != cursor
                        or contract.source_end > chunk_start
                    ):
                        state.poison(
                            "cached prefix does not end on an offline segment boundary"
                        )
                    self._consume_contract(
                        state,
                        contract=contract,
                        token_ids=cached_prefix[
                            contract.source_start : contract.source_end
                        ],
                    )
                    state.next_contract += 1
                    cursor = contract.source_end
                if state.next_position != chunk_start:
                    state.poison("cached prefix extent was not reconstructed exactly")
            self._install(key, state)
        else:
            self._states.move_to_end(key)
            if state.poisoned:
                raise ValueError("context token stream was previously poisoned")
            if state.input_verified:
                raise ValueError("context token stream was already completed")
            if (
                state.plan_digest != plan_digest
                or state.request_id != request_id
                or state.request_binding != request_binding
                or state.scheduler_total != scheduler_total
                or state.contracts != contracts
                or state.query_start != query_start
                or state.request_input_hash != request_input_hash
            ):
                state.poison("context plan/total changed across microforwards")
        return state

    @staticmethod
    def _validate_scheduler_extent(
        state: _ContextStreamState,
        *,
        positions: Tuple[int, ...],
        scheduler_current_extent: int,
    ) -> Tuple[int, int]:
        if type(scheduler_current_extent) is not int or scheduler_current_extent <= 0:
            state.poison("context certificate needs scheduler-owned current extent")
        start = positions[0]
        end = positions[-1] + 1
        if start != state.next_position:
            state.poison(
                f"context token stream gap/overlap: expected {state.next_position}, got {start}"
            )
        if end != scheduler_current_extent:
            state.poison("scheduler extent differs from the current contiguous rows")
        if end > state.scheduler_total:
            state.poison("context rows exceed scheduler-owned total tokens")
        return start, end

    @staticmethod
    def _consume_contract(
        state: _ContextStreamState,
        *,
        contract: ContextSegmentContract,
        token_ids: Tuple[int, ...],
    ) -> None:
        if state.cumulative_hash != contract.prefix_input_hash:
            state.poison("live cumulative prefix hash differs from snapshot contract")
        if token_ids_sha256(token_ids) != contract.token_hash:
            state.poison("live target token hash differs from snapshot contract")
        state.cumulative_hasher.update(_token_bytes(token_ids))
        state.next_position = contract.source_end
        if state.cumulative_hash != contract.full_input_hash:
            state.poison("live cumulative full-input hash differs from snapshot contract")

    def observe_snapshot_chunk(
        self,
        *,
        request_id: str,
        request_binding: Tuple[object, ...],
        plan: Mapping[str, object],
        positions: Sequence[int],
        token_ids: Sequence[int],
        scheduler_total: int,
        scheduler_current_extent: int,
    ) -> str:
        normalized_positions, normalized_tokens = _normalize_chunk(
            positions, token_ids
        )
        execution_profile = str(plan.get("mla_off_execution_profile", ""))
        head_scope_policy = str(plan.get("mla_off_head_scope_policy", ""))
        model_compat_hash = str(plan.get("model_compat_hash", ""))
        head_policy_hash = str(plan.get("head_policy_hash", ""))
        contract = ContextSegmentContract.from_mapping(
            plan,
            execution_profile=execution_profile,
            head_scope_policy=head_scope_policy,
            model_compat_hash=model_compat_hash,
            head_policy_hash=head_policy_hash,
        )
        if scheduler_total != contract.source_end:
            raise ValueError(
                "snapshot cumulative request must end exactly at source_end"
            )
        state = self._state(
            mode="snapshot",
            request_id=request_id,
            request_binding=request_binding,
            plan=plan,
            scheduler_total=scheduler_total,
            contracts=(contract,),
            query_start=contract.source_end,
            request_input_hash=contract.full_input_hash,
            chunk_start=normalized_positions[0],
        )
        start, end = self._validate_scheduler_extent(
            state,
            positions=normalized_positions,
            scheduler_current_extent=scheduler_current_extent,
        )
        if end <= contract.source_start:
            state.cumulative_hasher.update(_token_bytes(normalized_tokens))
            state.next_position = end
            if end == contract.source_start and (
                state.cumulative_hash != contract.prefix_input_hash
            ):
                state.poison("live prefix hash differs before target capture")
            return "prefix"
        if start != contract.source_start or end != contract.source_end:
            state.poison(
                "snapshot target must occupy one exact source_start/source_end microforward"
            )
        self._consume_contract(
            state, contract=contract, token_ids=normalized_tokens
        )
        state.next_contract = 1
        state.input_verified = True
        return "capture"

    def observe_restore_chunk(
        self,
        *,
        request_id: str,
        request_binding: Tuple[object, ...],
        plan: Mapping[str, object],
        positions: Sequence[int],
        token_ids: Sequence[int],
        scheduler_total: int,
        scheduler_current_extent: int,
        trusted_cached_prefix_tokens: Sequence[int] = (),
    ) -> str:
        normalized_positions, normalized_tokens = _normalize_chunk(
            positions, token_ids
        )
        execution_profile = str(plan.get("mla_off_execution_profile", ""))
        head_scope_policy = str(plan.get("mla_off_head_scope_policy", ""))
        model_compat_hash = str(plan.get("model_compat_hash", ""))
        head_policy_hash = str(plan.get("head_policy_hash", ""))
        offline_prefix_hash = _require_sha256(
            plan.get("offline_prefix_hash"), name="offline_prefix_hash"
        )
        request_input_hash = _require_sha256(
            plan.get("request_input_hash"), name="request_input_hash"
        )
        contracts = validate_context_segment_chain(
            plan.get("segments", ()),
            execution_profile=execution_profile,
            head_scope_policy=head_scope_policy,
            model_compat_hash=model_compat_hash,
            head_policy_hash=head_policy_hash,
            offline_prefix_hash=offline_prefix_hash,
        )
        query_start = plan.get("query_start")
        declared_total = plan.get("total_tokens")
        if type(query_start) is not int or query_start != contracts[-1].source_end:
            raise ValueError("restore query_start differs from context segment chain")
        if type(declared_total) is not int or declared_total != scheduler_total:
            raise ValueError("restore total_tokens differs from scheduler-owned total")
        if declared_total < query_start:
            raise ValueError("restore total_tokens ends inside the offline prefix")
        state = self._state(
            mode="restore",
            request_id=request_id,
            request_binding=request_binding,
            plan=plan,
            scheduler_total=scheduler_total,
            contracts=contracts,
            query_start=query_start,
            request_input_hash=request_input_hash,
            chunk_start=normalized_positions[0],
            trusted_cached_prefix_tokens=trusted_cached_prefix_tokens,
        )
        start, end = self._validate_scheduler_extent(
            state,
            positions=normalized_positions,
            scheduler_current_extent=scheduler_current_extent,
        )
        if state.next_contract < len(contracts):
            first_contract_index = state.next_contract
            cursor = start
            while state.next_contract < len(contracts) and cursor < end:
                contract = contracts[state.next_contract]
                if cursor != contract.source_start or contract.source_end > end:
                    state.poison(
                        "restore offline microforward must contain only complete "
                        "ordered source intervals"
                    )
                token_begin = contract.source_start - start
                token_end = contract.source_end - start
                self._consume_contract(
                    state,
                    contract=contract,
                    token_ids=normalized_tokens[token_begin:token_end],
                )
                state.next_contract += 1
                cursor = contract.source_end
            if cursor != end:
                state.poison(
                    "restore offline microforward ended outside a segment boundary"
                )
            consumed_contracts = state.next_contract - first_contract_index
            if consumed_contracts > 1:
                merged_tokens = plan.get("merged_prefill_tokens")
                radix_prefix_tokens = 0
                if plan.get("radix_prefix_role") == "consume":
                    candidate_prefix = plan.get("radix_prefix_tokens")
                    if (
                        isinstance(candidate_prefix, bool)
                        or not isinstance(candidate_prefix, int)
                        or candidate_prefix <= 0
                        or candidate_prefix != contracts[0].source_end
                        or plan.get("radix_prefix_input_hash")
                        != contracts[0].full_input_hash
                    ):
                        state.poison(
                            "cross-segment restore has an invalid radix-prefix origin"
                        )
                    radix_prefix_tokens = int(candidate_prefix)
                if (
                    isinstance(merged_tokens, bool)
                    or not isinstance(merged_tokens, int)
                    or int(merged_tokens) != end - start
                    or start < radix_prefix_tokens
                    or (start - radix_prefix_tokens) % int(merged_tokens) != 0
                ):
                    state.poison(
                        "cross-segment restore lacks its exact merged-prefill authorization"
                    )
            if state.next_contract == len(contracts):
                if state.cumulative_hash != offline_prefix_hash:
                    state.poison("live offline prefix hash differs after final segment")
                if query_start == scheduler_total:
                    if state.cumulative_hash != request_input_hash:
                        state.poison("live request hash differs at final offline row")
                    state.input_verified = True
            return "segment"

        if start < query_start:
            state.poison("restore suffix began before the certified offline prefix ended")
        state.cumulative_hasher.update(_token_bytes(normalized_tokens))
        state.next_position = end
        if end == scheduler_total:
            if state.cumulative_hash != request_input_hash:
                state.poison("live request hash differs at final input row")
            state.input_verified = True
            return "suffix_complete"
        return "suffix"

    def prepare_snapshot_publication(
        self,
        *,
        request_binding: Tuple[object, ...],
        confirmation_digest: str,
        seg_hash: str,
        model_compat_hash: str,
        head_policy_hash: str,
        generation_id: str,
        published_layer_ids: Tuple[int, ...],
    ) -> PreparedContextSnapshotPublication:
        confirmation_digest = _require_sha256(
            confirmation_digest, name="confirmation_digest"
        )
        state = self._states.get(("snapshot", request_binding))
        if state is None or not state.input_verified or state.poisoned:
            raise ValueError(
                "snapshot publication has no live verified input certificate"
            )
        if state.publication_receipt is not None:
            raise RuntimeError("snapshot publication receipt was already installed")
        if len(state.contracts) != 1:
            raise RuntimeError("snapshot publication input contract is not singular")
        contract = state.contracts[0]
        if (
            seg_hash != contract.seg_hash
            or model_compat_hash != contract.model_compat_hash
            or head_policy_hash != contract.head_policy_hash
        ):
            raise ValueError("snapshot publication identity differs from verified input")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError("snapshot publication generation_id must be non-empty")
        if published_layer_ids != tuple(range(3, 40)):
            raise ValueError("snapshot publication must cover exactly layers 3..39")
        encoded = json.dumps(
            {
                "confirmation_digest": confirmation_digest,
                "generation_id": generation_id,
                "head_policy_hash": head_policy_hash,
                "model_compat_hash": model_compat_hash,
                "published_layer_ids": list(published_layer_ids),
                "request_binding": list(request_binding),
                "seg_hash": seg_hash,
                "schema": "redknot_context_snapshot_publication_v1",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        receipt = SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()
        return PreparedContextSnapshotPublication(_state=state, receipt=receipt)

    def poison_snapshot_publication(
        self,
        *,
        request_binding: Tuple[object, ...],
    ) -> None:
        state = self._states.get(("snapshot", request_binding))
        if state is not None:
            state.poisoned = True

    def snapshot(
        self,
        *,
        mode: str,
        request_binding: Tuple[object, ...],
    ) -> Mapping[str, object]:
        state = self._states[(mode, request_binding)]
        return {
            "mode": state.mode,
            "request_id": state.request_id,
            "request_binding": state.request_binding,
            "next_position": state.next_position,
            "next_contract": state.next_contract,
            "cumulative_hash": state.cumulative_hash,
            "input_verified": state.input_verified,
            "publication_confirmed": state.publication_receipt is not None,
            "publication_receipt": state.publication_receipt,
            "poisoned": state.poisoned,
        }


__all__ = [
    "CONTEXT_SEGMENT_SCHEMA",
    "ContextSegmentContract",
    "ContextTokenStreamRegistry",
    "EMPTY_TOKEN_IDS_SHA256",
    "NATIVE_FULL_SCOPE_POLICY",
    "PreparedContextSnapshotPublication",
    "context_segment_sha256",
    "token_ids_sha256",
    "validate_context_segment_chain",
]
