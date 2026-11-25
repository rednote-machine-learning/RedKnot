"""Private extension of the frozen all-layer tail proof for physical Cone15/V2 rows."""
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

from cone15_plan import Cone15Plan


def require(ok,message):
    if not ok:
        raise ValueError(message)


def live_covered(event, wanted, layer, field, pos_key, *, minimal_plan=None, two_stage_state=None):
    selected=event['selected_positions']
    if 'backfill_positions' in event:
        if two_stage_state is None:
            return False
        mid=two_stage_state['mid']
        affected=two_stage_state['affected']
        states=two_stage_state['states']
        updated=event['backfill_positions']
        if (layer,'hidden') not in states.data or (layer,'swa') not in states.data:
            return False
        return (3<=layer<mid and updated==[v for v in selected if v in affected[layer]]
                and event[field]==len(updated)
                and all(p in states.index for p in selected if p not in affected[layer])
                and all(p in updated or p in states.index for p in wanted))
    if pos_key not in event:
        return event[field]==event['selected_rows']
    live=event[pos_key]
    if minimal_plan is not None and 21<=layer<=39:
        a,z=event['chunk_start'],event['chunk_end']
        geom=minimal_plan.group(layer,a,z)
        if pos_key.endswith('mhc_positions'):
            expected_inputs=set(geom.inputs)
            actual_inputs=set(event['cone15_input_positions'])
            selected_set=set(selected)
            inputs_ok=(actual_inputs==expected_inputs) or (not expected_inputs and actual_inputs==selected_set)
            actual=set(live)
            extra=actual-actual_inputs
            return (inputs_ok and live==sorted(actual) and expected_inputs<=actual<=selected_set
                    and len(live)==event[field] and len(extra)<=1)
        return list(geom.queries)==live and event[field]==len(live)
    return (21<=layer<=39 and live==sorted(set(live)) and set(wanted)<=set(live)<=set(selected)
            and event[field]==len(live))


def private_tail_proof(base):
    import index20_runtime as source
    cls=next(c for c in base.__mro__ if c.__qualname__=='index20_controller_class.<locals>.Index20Controller')
    require(cls.__dict__['finish'].__globals__ is source.__dict__, 'Expected newly constructed Index20 controller type')
    src=textwrap.dedent(inspect.getsource(source.mandatory_tail_update_proof))
    for field,pos_key in [('actual_ffn_rows','cone15_ffn_positions'),('actual_attention_rows','cone15_core_positions'),
                          ('actual_residual_rows','cone15_mhc_positions'),('actual_mhc_rows','cone15_mhc_positions')]:
        old='e["'+field+'"]==e["selected_rows"]'
        require(src.count(old)==1,'Unrecognized native row check')
        src=src.replace(old,'physical_rows(e,wanted,layer,'+repr(field)+','+repr(pos_key)+')')
    active=[None]
    independent_plans={}

    def physical_rows(event,wanted,layer,field,pos_key):
        ctrl=active[0]
        minimal_plan=None
        two_stage_state=None
        if ctrl is not None:
            two_stage_state=getattr(ctrl,'_v2_two_stage_state',None)
            policy=getattr(ctrl,'_v2_policy',None)
            if policy is not None and policy.minimal_cone and layer>=21 and pos_key in event:
                current=getattr(ctrl,'_cone_plan',None)
                if current is not None:
                    key=id(current)
                    if key not in independent_plans:
                        independent_plans[key]=Cone15Plan(
                            current.total,current.selected,(),last_frame_start=current.last_frame_start)
                    minimal_plan=independent_plans[key]
        return live_covered(event,wanted,layer,field,pos_key,
                            minimal_plan=minimal_plan,two_stage_state=two_stage_state)

    ns=dict(source.__dict__,physical_rows=physical_rows)
    exec(compile(src,'<private Cone15 native tail proof>','exec'),ns)
    orig=cls.__dict__['finish']
    inner=FunctionType(orig.__code__,dict(orig.__globals__,mandatory_tail_update_proof=ns['mandatory_tail_update_proof']),
                       orig.__name__,orig.__defaults__,orig.__closure__)
    inner.__kwdefaults__=orig.__kwdefaults__

    def finish_with_proof_context(self):
        prev=active[0]
        active[0]=self
        independent_plans.clear()
        try:
            return inner(self)
        finally:
            active[0]=prev
            independent_plans.clear()

    cls.finish=finish_with_proof_context
