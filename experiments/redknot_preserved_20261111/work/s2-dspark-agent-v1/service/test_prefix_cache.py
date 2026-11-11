"""CPU control/ownership regressions, never a substitute for GPU parity."""
import contextlib
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.engine_bridge import TokenRequest
from service.prefix_cache import PrefixCache, ScheduleObserver, checkpoint_cuts


class Tensor:
    def __init__(self, data):
        self.data, self._version, self.device = data, 0, 'cpu-fixture'
    @property
    def shape(self):
        return (len(self.data), len(self.data[0])) if self.data and isinstance(self.data[0], list) else (len(self.data),)
    def element_size(self): return 1
    def __getitem__(self, index): return Tensor(self.data[index])
    def clone(self): return Tensor(copy.deepcopy(self.data))
    def index_select(self, dim, indices): return Tensor([list(self.data[i]) for i in indices])
    def index_copy_(self, dim, indices, source):
        for i, values in zip(indices, source.data): self.data[i][:] = values
        self._version += 1
    def copy_(self, source):
        self.data[:] = source.data
        self._version += 1


class Event:
    def __init__(self, log): self.log = log
    def record(self): self.log.append('copy_recorded')
    def synchronize(self): self.log.append('copy_synchronized')


class Seq:
    def __init__(self, rid, log, ids):
        self.request_id, self.log = rid, log
        self._cached, self.cached_hit_tokens, self.swa_valid_start = 0, 0, 0
        self.ids, self.previous_num_tokens = ids,0
    @property
    def num_cached_tokens(self): return self._cached
    @num_cached_tokens.setter
    def num_cached_tokens(self, value):
        self.log.append(('cached',value)); self._cached = value;self.previous_num_tokens=value


class PrefixTests(unittest.TestCase):
    def fixture(self, cap=64):
        log=[]
        storage={'gpu_blocks':1,'gpu_states':2,'mtp_info_pool.token_ids':3,'mtp_info_pool.logprobs':4,
                 'main_graph.fixture':5,'dflash_graph.fixture':6}
        for name,value in (('inspect_native_decode_storage',dict(graph_storage=storage)),('native_graph_storage',storage)):
            mocked=patch('service.prefix_cache.'+name,return_value=value)
            mocked.start();self.addCleanup(mocked.stop)
        ce=NS(gpu_blocks=Tensor([[i]*4 for i in range(12)]),gpu_states=Tensor([[0]*4,[11]*4]))
        engine=NS(cache_config=NS(block_size=128),driver_worker=NS(cache_engine=ce),multi_modal_processor=None)
        engine.add_request=lambda **kw:Seq(kw['request_id'],log,kw['prompt_token_ids'])
        engine.scheduler=NS(try_allocate_seq=lambda seq:True,add_sequence=lambda seq:log.append('enqueued'),
                            free_seq=lambda seq:log.append('freed'))
        def descriptor(seq):
            return dict(state_id=1,block_table=[4,5,6,7],request_id=seq.request_id,
                num_cached_tokens=seq.num_cached_tokens,previous_num_tokens=seq.previous_num_tokens,
                swa_valid_start=seq.swa_valid_start,block_size=128,last_copy_state_token_idx=0,
                prompt_tokens=len(seq.ids),output_tokens=0,schedule_status='WAITING',total_inflight_tokens=0,
                last_inflight_tokens=0,mtp_info_slot=0,mtp_has_draft_info=False,mtp_last_output_token=seq.ids[-1],
                mtp_accepted_count=0,mtp_draft_count=0,mtp_accept_rate_count=0,mtp_prev_accept_rate_count=0,
                mtp_confidence_count=0,mtp_prev_confidence_count=0,mtp_process_count=0,mtp_schedule_count=0)
        bridge=NS(sequence_descriptor=descriptor,descriptor_contract='fresh-sequence-dspark-prefix-v1')
        torch=NS(long='long',tensor=lambda ids,**kw:list(ids),cuda=NS(Event=lambda:Event(log)))
        cache=PrefixCache(engine,bridge,torch,contextlib.nullcontext,cap)
        port=NS(params_factory=lambda **kw:kw,array_factory=lambda ids:list(ids))
        return cache,ce,port,log

    def request(self, rid, ids=tuple(range(300))):
        return TokenRequest(rid,ids,32,'prefix',0,documents=(dict(input_end=256),))

    def capture(self, cache, n=256):
        seq=Seq(cache.active.request_id,[],cache.active.prompt_token_ids);seq.num_cached_tokens=n
        cache.capture(dict(is_prompt=True,is_placeholder=False,request_id=cache.active.request_id,
            seqlen_q=n,seqlen_k=n,block_table=list(range(1,1+(n+127)//128)),state_id=1,swa_valid_start=0),seq)

    def test_only_actual_aligned_natural_and_native_checkpoints(self):
        r=TokenRequest('x',tuple(range(17000)),32,'prefix',0,
            documents=(dict(input_end=100),dict(input_end=333),dict(input_end=9999)))
        self.assertEqual(checkpoint_cuts(r),[256,8192,9984,16384])

    def test_snapshot_owns_both_bytes_and_restore_completes_before_cached_counter(self):
        cache,ce,port,log=self.fixture()
        cold=self.request('cold');self.assertEqual(cache.begin(cold),0)
        self.capture(cache);receipt=cache.finish()['captures'][0]
        ce.gpu_blocks.data[1][:]=[99]*4;ce.gpu_states.data[1][:]=[88]*4
        hit=self.request('hit',tuple(range(299))+(987,))
        self.assertEqual(cache.begin(hit),256)
        log.clear();seq=cache.submit(hit,port)
        self.assertEqual(ce.gpu_blocks.data[4:6],[[1]*4,[2]*4])
        self.assertEqual(ce.gpu_states.data[1],[11]*4)
        self.assertLess(max(i for i,v in enumerate(log) if v=='copy_synchronized'),log.index(('cached',256)))
        self.assertLess(log.index(('cached',256)),log.index('enqueued'))
        report=cache.finish()
        self.assertEqual(report['restore']['snapshot_id'],receipt['snapshot_id'])
        self.assertEqual(seq.num_cached_tokens,256)
        self.assertTrue(report['cache_hit'])

    def test_changed_prefix_and_insufficient_suffix_are_misses(self):
        cache,ce,port,log=self.fixture();cache.begin(self.request('cold'));self.capture(cache);cache.finish()
        changed=list(range(300));changed[10]=777
        self.assertEqual(cache.begin(self.request('changed',tuple(changed))),0);cache.finish()
        self.assertEqual(cache.begin(self.request('too-short',tuple(range(257)))),0);cache.finish()
        self.assertEqual(cache.begin(self.request('disabled'),enabled=False),0);cache.finish()

    def test_compact_near_tail_requires_equal_total_length_or_earlier_checkpoint(self):
        cache,ce,port,log=self.fixture(cap=64)
        cold=TokenRequest('cold',tuple(range(300)),32,'prefix',0,documents=(dict(input_end=128),dict(input_end=256)))
        cache.begin(cold);self.capture(cache,n=128);self.capture(cache,n=256);cache.finish()
        target=self.request('shorter',tuple(range(299)))
        self.assertEqual(cache.begin(target),128)
        report=cache.finish()
        self.assertEqual(report['incompatible_exact_prefix_matches'][0]['prefix_tokens'],256)
        self.assertEqual(report['incompatible_exact_prefix_matches'][0]['reason'],'near_tail_compact140_cutoff_differs')
        self.assertEqual(cache.begin(self.request('same-length')),256);cache.finish()

    def test_final_prompt_checkpoint_never_captures_proposal_state(self):
        r=self.request('aligned',tuple(range(256)))
        self.assertEqual(checkpoint_cuts(r),[])
        cache,ce,port,log=self.fixture();cache.begin(r)
        self.capture(cache,n=256)
        self.assertEqual(cache.finish()['captures'],[])

    def test_stale_proposal_and_native_storage_replacement_fail_closed(self):
        cache,ce,port,log=self.fixture();cache.begin(self.request('cold'));self.capture(cache);cache.finish()
        real=cache.bridge.sequence_descriptor
        cache.bridge.sequence_descriptor=lambda seq:dict(real(seq),mtp_has_draft_info=True)
        hit=self.request('hit');cache.begin(hit)
        with self.assertRaisesRegex(ValueError,'proposal'):
            cache.submit(hit,port)
        self.assertIn('freed',log)
        cache.active=None
        with patch('service.prefix_cache.native_graph_storage',return_value={'gpu_blocks':99}):
            with self.assertRaisesRegex(ValueError,'inventory'):
                cache.begin(self.request('replacement'))

    def test_uncommitted_state_rejected_and_budget_evicts_owned_snapshots(self):
        cache,ce,port,log=self.fixture(cap=12);cache.begin(self.request('one'))
        with self.assertRaisesRegex(ValueError,'committed'):
            cache.capture(dict(is_prompt=True,is_placeholder=False,request_id='one',seqlen_k=256),NS(num_cached_tokens=128))
        self.capture(cache);first=cache.finish()['captures'][0]['snapshot_id']
        second=list(range(300));second[0]=909
        cache.begin(self.request('two',tuple(second)));self.capture(cache)
        report=cache.finish();self.assertEqual(report['evictions'],[first]);self.assertEqual(cache.bytes,12)
        cache.clear();self.assertEqual(cache.bytes,0);self.assertFalse(cache.bank)

    def test_meter_counts_real_rows_and_restores_prior_owned_hook(self):
        class Engine:
            def _execute_model_async_step(self, seqs, outputs): return 'native'
        engine=Engine();calls=[]
        batch=lambda seqs,outputs:calls.append('batch')
        engine._execute_model_async_step=batch
        meter=ScheduleObserver(engine,0);r=self.request('meter');meter.configure(r,128)
        metadata=NS(request_id='meter',seqlen_k=300,seqlen_q=172,is_placeholder=False)
        engine._execute_model_async_step([metadata],NS(prompt_run=True))
        report=meter.finish();self.assertEqual(report['scheduled_prefill_rows'],172)
        self.assertEqual(report['first_prefill_q_start'],128);self.assertEqual(calls,['batch'])
        meter.close();self.assertIs(engine.__dict__['_execute_model_async_step'],batch)
        del engine._execute_model_async_step
        idle=ScheduleObserver(engine,7);idle.configure(r)
        engine._execute_model_async_step([NS(is_placeholder=True)],NS(prompt_run=True))
        self.assertEqual(idle.finish()['prefill_ranges'],[]);idle.close()
        self.assertNotIn('_execute_model_async_step',engine.__dict__)


if __name__=='__main__':unittest.main()
