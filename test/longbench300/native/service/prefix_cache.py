"""Bounded GPU-copy prefix cache using exact native prefill checkpoints.

This is an explicit external system adapter, not an automatic native radix
cache. Native kernels/layouts stay untouched. Each checkpoint copies both KV
blocks and the complete raw mutable state at the same completed prompt step.
"""
from collections import OrderedDict
from math import ceil
import time
from .engine_bridge import token_digest
from .native_runtime import require, seal, digest
from prefix_decode_contract import (inspect_native_decode_storage, native_graph_storage,
    validate_fresh_prefix_resume, validate_stable_graph_storage, compact_prefix_compatible)


def checkpoint_cuts(request):
    n = len(request.prompt_token_ids)
    natural = [(doc['input_end']//128)*128 for doc in request.documents]
    # Do not snapshot the final prompt step: it may already have generated a
    # DFlash proposal. A reusable snapshot is exclusively an intermediate
    # prefill boundary with at least two unprocessed native suffix tokens.
    return sorted({v for v in natural+list(range(8192,n+1,8192)) if 128 <= v <= n-2 and v%128 == 0})


class ScheduleObserver:
    """Actual scheduled query rows, identical observation hook in all arms."""
    def __init__(self, engine, rank):
        self.engine, self.rank = engine, rank
        self.previous_override = engine.__dict__.get('_execute_model_async_step')
        self.original = engine._execute_model_async_step
        self.request = None
        self.prefix = None
        def hook(seqs, outputs):
            prompt = bool(outputs.prompt_run)
            frames = [dict(request_id=s.request_id,q_start=s.seqlen_k-s.seqlen_q,
                q_end=s.seqlen_k,rows=s.seqlen_q) for s in seqs if not s.is_placeholder]
            if self.request is not None:
                for frame in frames:
                    require(self.rank == 0 and frame['request_id'] == self.request.request_id,
                            'foreign actual native scheduled request')
                if prompt:
                    self.ranges.extend(frames)
                else:
                    self.decode_steps += bool(frames)
            descriptors = None
            if self.prefix is not None and prompt and frames:
                descriptors = [self.prefix.bridge.metadata_descriptor(s) for s in seqs if not s.is_placeholder]
            result = self.original(seqs,outputs)
            # Native Worker promptly postprocesses current prefill (including
            # sample_end synchronization); decode under ZOS is not a checkpoint.
            if descriptors:
                require(len(descriptors) == len(outputs.scheduled_seqs) == 1, 'serial prefix checkpoint expected')
                self.prefix.capture(descriptors[0], outputs.scheduled_seqs[0])
            return result
        self.hook = hook
        engine._execute_model_async_step = hook

    def configure(self, request, cached=0):
        require(self.request is None, 'schedule observer previous phase unfinished')
        self.request, self.cached, self.ranges, self.decode_steps = request, cached, [], 0

    def finish(self):
        r = self.request
        require(r is not None, 'no active schedule measurement')
        position = self.cached
        for frame in self.ranges:
            require(frame['q_start'] == position and frame['q_end']-frame['q_start'] == frame['rows'],
                    'actual native prefill is not a contiguous suffix')
            position = frame['q_end']
        if self.rank == 0:
            require(position == len(r.prompt_token_ids), 'native prefill omitted actual query rows')
        else:
            require(not self.ranges and self.decode_steps == 0, 'inactive DP rank ran real rows')
        value = seal(dict(schema='s2-native-scheduled-rows-v1',rank=self.rank,request_id=r.request_id,
            input_tokens=len(r.prompt_token_ids), initial_cached_tokens=self.cached if self.rank == 0 else 0,
            scheduled_prefill_rows=sum(f['rows'] for f in self.ranges),
            first_prefill_q_start=None if not self.ranges else self.ranges[0]['q_start'],
            prefill_ranges=list(self.ranges),decode_steps=self.decode_steps))
        self.request = None
        return value

    def close(self):
        require(self.engine.__dict__.get('_execute_model_async_step') is self.hook,'native meter hook ownership changed')
        if self.previous_override is None:
            del self.engine._execute_model_async_step
        else:
            self.engine._execute_model_async_step = self.previous_override


class PrefixCache:
    def __init__(self, engine, bridge, torch, allow_sync, cap):
        self.engine, self.bridge, self.t, self.allow_sync, self.cap = engine, bridge, torch, allow_sync, cap
        self.cache_engine = engine.driver_worker.cache_engine
        self.bank, self.bytes, self.generation = OrderedDict(), 0, 0
        self.active = None
        require(getattr(bridge,'descriptor_contract',None) == 'fresh-sequence-dspark-prefix-v1',
                'Prefix D-Spark requires the new read-only descriptor bridge')
        self.decode_contract = seal(inspect_native_decode_storage(engine))

    def _check_storage(self):
        return validate_stable_graph_storage(self.decode_contract['graph_storage'],native_graph_storage(self.engine))

    def clear(self):
        require(self.active is None,'cannot clear active prefix cache')
        self.bank.clear();self.bytes=0

    def begin(self, request, *, diagnostic=False, enabled=True):
        require(self.active is None, 'overlapping prefix request')
        self._check_storage()
        self.active, self.diagnostic, self.enabled = request, diagnostic, enabled
        self.cuts, self.captures, self.evictions = checkpoint_cuts(request), [], []
        self.incompatible_matches=[]
        self.bytes_before, self.selected, self.restore = self.bytes, None, None
        if enabled:
            for key, entry in reversed(self.bank.items()):
                n = entry['receipt']['prefix_tokens']
                if n <= len(request.prompt_token_ids)-2 and tuple(request.prompt_token_ids[:n]) == entry['ids']:
                    if not compact_prefix_compatible(n,entry['receipt']['source_prompt_tokens'],len(request.prompt_token_ids)):
                        self.incompatible_matches.append(dict(snapshot_id=entry['receipt']['snapshot_id'],prefix_tokens=n,
                            source_prompt_tokens=entry['receipt']['source_prompt_tokens'],target_prompt_tokens=len(request.prompt_token_ids),
                            reason='near_tail_compact140_cutoff_differs'))
                        continue
                    if self.selected is None or n > self.selected['receipt']['prefix_tokens']:
                        self.selected = entry
            if self.selected:
                key = self.selected['key'];self.bank.move_to_end(key)
        return 0 if self.selected is None else self.selected['receipt']['prefix_tokens']

    def _fingerprint(self, tensor):
        import hashlib
        with self.allow_sync():
            raw = tensor.detach().contiguous().cpu().numpy().tobytes()
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(raw).hexdigest())

    def _validate_entry(self, entry):
        require(entry['blocks']._version == entry['blocks_version']
            and entry['state']._version == entry['state_version'], 'immutable prefix snapshot was modified')
        entry['event'].synchronize()

    def submit(self, request, port):
        require(self.active is request, 'prefix submission lost request ownership')
        params = port.params_factory(temperature=request.temperature,top_p=request.top_p,top_k=-1,
            max_tokens=request.max_tokens,ignore_eos=False,seq_seed=request.seed)
        mm = dict(image_bytes=[],image_sizes=[]) if self.engine.multi_modal_processor is not None else None
        seq = self.engine.add_request(user_id='s2-agent',request_id=request.request_id,sampling_params=params,
            prompt_token_ids=port.array_factory(request.prompt_token_ids),multi_modal_data=mm,
            enqueue=False,chunk_checkpoints=self.cuts if self.enabled else None)
        require(seq is not None and seq.num_cached_tokens == 0,'new prefix sequence must start uncached')
        allocated = False
        try:
            entry = self.selected
            if entry is not None:
                with self.allow_sync():
                    self._validate_entry(entry)
                    require(self.engine.scheduler.try_allocate_seq(seq), 'native block/state allocation unavailable for prefix copy')
                    allocated = True
                    dst = self.bridge.sequence_descriptor(seq)
                    receipt = entry['receipt'];n = receipt['prefix_tokens']
                    count = len(receipt['source_block_ids'])
                    require(dst['state_id'] > 0 and len(dst['block_table']) >= count, 'native prefix destination geometry differs')
                    dst_ids = dst['block_table'][:count]
                    index = self.t.tensor(dst_ids,device=self.cache_engine.gpu_blocks.device,dtype=self.t.long)
                    self.cache_engine.gpu_blocks.index_copy_(0,index,entry['blocks'])
                    self.cache_engine.gpu_states[dst['state_id']].copy_(entry['state'])
                    copied = self.t.cuda.Event();copied.record();copied.synchronize()
                    fingerprints = None
                    if self.diagnostic:
                        fingerprints = dict(kv=self._fingerprint(self.cache_engine.gpu_blocks.index_select(0,index)),
                            state=self._fingerprint(self.cache_engine.gpu_states[dst['state_id']]))
                        require(fingerprints == receipt['raw_fingerprints'],'restored prefix bytes differ before native continuation')
                    # Advance native cache accounting only after actual complete
                    # state/KV transfer; no arbitrary scalar-only prefix injection.
                    seq.swa_valid_start = receipt['swa_valid_start']
                    seq.num_cached_tokens = n
                    seq.cached_hit_tokens = n
                    after = self.bridge.sequence_descriptor(seq)
                    checkpoint = dict(is_prompt=True,is_placeholder=False,
                        seqlen_q=receipt['metadata_seqlen_q'],seqlen_k=n,
                        committed_num_cached_tokens=receipt['committed_num_cached_tokens'],
                        swa_valid_start=receipt['swa_valid_start'],source_prompt_tokens=receipt['source_prompt_tokens'])
                    admission = validate_fresh_prefix_resume(request.prompt_token_ids,entry['ids'],checkpoint,dst,after)
                    stable = self._check_storage()
                    self.restore = dict(snapshot_id=receipt['snapshot_id'],prefix_tokens=n,
                        prefix_ids_sha256=receipt['prefix_ids_sha256'],capture_request_id=receipt['capture_request_id'],
                        capture_metadata_seqlen_k=receipt['metadata_seqlen_k'],
                        capture_committed_num_cached_tokens=receipt['committed_num_cached_tokens'],
                        source_block_ids=receipt['source_block_ids'],destination_block_ids=dst_ids,
                        source_state_id=receipt['source_state_id'],destination_state_id=dst['state_id'],
                        kv_bytes=receipt['kv_bytes'],state_bytes=receipt['state_bytes'],copy_complete=True,
                        raw_state_complete=True,raw_kv_and_state_same_checkpoint=True,
                        restored_num_cached_tokens=seq.num_cached_tokens,source_fingerprints=receipt['raw_fingerprints'],
                        restored_fingerprints=fingerprints,
                        fresh_sequence_before=dst,fresh_sequence_after=after,
                        fresh_resume_admission=admission,graph_storage_stability=stable,
                        decode_storage_contract_sha256=self.decode_contract['sha256'])
            self.engine.scheduler.add_sequence(seq)
            if self.engine.multi_modal_processor is not None:
                self.engine.multi_modal_processor.add(request.request_id,mm,num_cached_tokens=seq.num_cached_tokens)
            return seq
        except BaseException:
            if allocated:self.engine.scheduler.free_seq(seq)
            raise
        finally:
            # The independent native destination now owns its bytes. No stale
            # Python reference may keep an evicted bank entry outside the cap.
            self.selected = None

    def capture(self, descriptor, seq):
        if self.active is None or not self.enabled:
            return
        n = descriptor['seqlen_k']
        if n not in self.cuts:
            return
        require(descriptor['is_prompt'] is True and descriptor['is_placeholder'] is False
            and descriptor['request_id'] == self.active.request_id and seq.num_cached_tokens == n,
            'checkpoint must be an actual committed current prefill boundary')
        committed = self.bridge.sequence_descriptor(seq)
        require(n <= len(self.active.prompt_token_ids)-2 and committed['prompt_tokens'] == len(self.active.prompt_token_ids)
                and committed['output_tokens'] == 0
                and committed['mtp_has_draft_info'] is False and committed['mtp_draft_count'] == 0,
                'Prefix capture must precede final-prompt sampling/proposal production')
        self._check_storage()
        # Near-tail consumer/DFlash state is cutoff-dependent, so equal IDs
        # from different complete prompt lengths are distinct snapshots.
        key = (n,token_digest(self.active.prompt_token_ids[:n]),len(self.active.prompt_token_ids))
        if key in self.bank:
            return
        count = ceil(n/self.engine.cache_config.block_size)
        block_ids = descriptor['block_table'][:count]
        state_id = descriptor['state_id']
        require(len(block_ids) == count and state_id > 0,'checkpoint lacks owned native KV/state')
        ce = self.cache_engine
        kv_bytes = count*ce.gpu_blocks.shape[1]*ce.gpu_blocks.element_size()
        state_bytes = ce.gpu_states.shape[1]*ce.gpu_states.element_size()
        cost = kv_bytes+state_bytes
        if cost > self.cap:
            self.captures.append(dict(prefix_tokens=n,skipped='snapshot_exceeds_budget',required_bytes=cost))
            return
        while self.bank and self.bytes+cost > self.cap:
            evicted, old = self.bank.popitem(last=False)
            self.bytes -= old['bytes'];self.evictions.append(old['receipt']['snapshot_id'])
            del old
        with self.allow_sync():
            index = self.t.tensor(block_ids,device=ce.gpu_blocks.device,dtype=self.t.long)
            # Same stream and same completed prompt checkpoint. Both snapshots
            # finish before returning a first token or advancing the engine.
            blocks = ce.gpu_blocks.index_select(0,index)
            state = ce.gpu_states[state_id].clone()
            event = self.t.cuda.Event();event.record();event.synchronize()
            raw = dict(kv=self._fingerprint(blocks),state=self._fingerprint(state)) if self.diagnostic else None
        self.generation += 1
        receipt = dict(snapshot_id=digest(dict(prefix=key,generation=self.generation,request=self.active.request_id)),
            prefix_tokens=n,prefix_ids_sha256=key[1],capture_request_id=self.active.request_id,is_prompt=True,
            metadata_seqlen_q=descriptor['seqlen_q'],metadata_seqlen_k=n,committed_num_cached_tokens=seq.num_cached_tokens,
            source_block_ids=block_ids,source_state_id=state_id,swa_valid_start=descriptor['swa_valid_start'],
            source_prompt_tokens=len(self.active.prompt_token_ids),compact_keep_tokens=140,
            kv_bytes=kv_bytes,state_bytes=state_bytes,copy_complete=True,raw_state_complete=True,
            raw_kv_and_state_same_checkpoint=True,raw_fingerprints=raw,
            committed_sequence=committed,decode_storage_contract_sha256=self.decode_contract['sha256'],
            prefix_only_intermediate_checkpoint=True,old_proposal_pool_copied=False)
        entry = dict(key=key,ids=tuple(self.active.prompt_token_ids[:n]),blocks=blocks,state=state,event=event,
            blocks_version=blocks._version,state_version=state._version,bytes=cost,receipt=receipt)
        self.bank[key] = entry;self.bytes += cost;self.captures.append(receipt)
        require(self.bytes <= self.cap,'prefix bank exceeded actual memory cap')

    def finish(self):
        r = self.active
        require(r is not None,'no active prefix request')
        restore = self.restore
        stable = self._check_storage()
        value = seal(dict(schema='s2-prefix-copy-request-v1',method='Prefix(copy)',request_id=r.request_id,
            input_tokens=len(r.prompt_token_ids),input_ids_sha256=token_digest(r.prompt_token_ids),
            cache_hit=restore is not None,reused_prefix_tokens=0 if restore is None else restore['prefix_tokens'],
            matched_prefix_sha256=None if restore is None else restore['prefix_ids_sha256'],restore=restore,
            captures=self.captures,bank_bytes_before=self.bytes_before,bank_bytes_after=self.bytes,
            bank_budget_bytes=self.cap,evictions=self.evictions,eviction_policy='whole_snapshot_LRU',
            snapshot_policy='128_aligned_natural_body_ends_and_8192_token_checkpoints',
            checkpoint_cuts=self.cuts if self.enabled else [],copy_cost_included=True,
            independent_same_checkpoint_raw_kv_and_state=True,zero_copy_claim=False,
            native_extra_state_slots_required=False,diagnostic_fingerprints=self.diagnostic,
            decode_storage_contract=self.decode_contract,graph_storage_stability=stable,
            proposal_policy='fresh_native_slot_then_native_suffix_and_decode',
            old_proposal_pool_copied=False,prefix_checkpoint_only=True,
            incompatible_exact_prefix_matches=self.incompatible_matches,
            compact_prefix_policy='before_target_cutoff_or_identical_source_target_prompt_length',compact_keep_tokens=140))
        self.active = None
        return value
