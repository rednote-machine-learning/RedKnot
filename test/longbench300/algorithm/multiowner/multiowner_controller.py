"""Private owner binding/padded EP flush around the unchanged S1.3 controller."""
from world_plan_coordinator import require,seal,validate_world_plan,validate_installation


def multiowner_controller_class(baseclass):
    class MultiOwnerController(baseclass):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.multiowner_worker=None;self.multiowner_spec=None
            self.multiowner_installation=None;self.multiowner_padding=[]

        def reset(self,*args,**kwargs):
            # A new native request must receive a new world binding. Never let
            # a prior candidate's agreement affect capture or qualification.
            self.multiowner_spec=None;self.multiowner_installation=None;self.multiowner_padding=[]
            return super().reset(*args,**kwargs)

        def bind_multiowner(self,spec,receipts):
            validate_world_plan(spec)
            worker=self.multiowner_worker
            require(worker is not None and worker.spec==spec,'Owner controller lacks its installed world worker')
            owner=next((o for o in spec['owners']if o['rank']==worker.rank),None)
            require(owner is not None and self.state=='RUNNING' and self.coverage==0
                and owner['request_id']==self.rid and owner['input_tokens']==len(self.ids)
                and owner['input_ids_sha256']==self.input_sha,'Owner controller/input differs from world request')
            require(owner['schedule']==self.batch_schedule,'Local controller schedule differs from world plan')
            proof=validate_installation(spec,receipts)
            if self.index20_enabled:
                require(spec['execution_kind']=='index20' and self._index20_installation is None,
                    'Candidate installation duplicated or native-only')
                # This is a new explicit per-owner installation proof, not a
                # fabricated rank-zero receipt for all eight workers.
                self._index20_installation=proof
            else:
                require(spec['execution_kind']=='native' and self.batch_schedule is None,
                    'Initial multi-owner prototype does not admit S1.2 group17 schedules')
            self.multiowner_spec=spec;self.multiowner_installation=proof;self.multiowner_padding=[]
            return proof

        def _run_deep_group(self,model,caches,group):
            result=super()._run_deep_group(model,caches,group)
            if self.multiowner_spec is None or group.end!=len(self.ids):return result
            worker=self.multiowner_worker;agreement=worker.flush_agreement
            require(agreement is not None,'Owner final deep group has no world agreement')
            local=agreement['local_groups_by_rank'][worker.rank]
            require(len(self._index20_groups)==local and not self.multiowner_padding,
                'Owner group count differs from the agreed actual selection')
            extra=agreement['world_groups']-local
            layers=worker.run_empty_deep(group.full_meta,extra)
            self.multiowner_padding.append(dict(local_groups=local,world_groups=agreement['world_groups'],
                empty_groups=extra,layers=layers,actual_empty_ep_calls=len(layers),
                selected_hidden_changed=False,kv_projected=False,lastkv_reprojected=False))
            return result

        def finish(self):
            report=super().finish()
            if self.multiowner_spec is not None:
                worker=self.multiowner_worker
                report['multiowner']=seal(dict(schema='s13-multiowner-controller-proof-v1',
                    owner_rank=worker.rank,request_id=self.rid,world_plan_sha256=self.multiowner_spec['sha256'],
                    installation=self.multiowner_installation,padding=list(self.multiowner_padding),
                    local_request_count=1,first3_and_native_indexer_per_owner=True,
                    lastkv_bank_request_local=True,owner_bank_shared_with_other_requests=False,
                    gpu_batch_quality_or_speedup_claim=False))
                self.last_report=report
            return report
    return MultiOwnerController
