"""Private extension of the frozen all-layer tail proof for physical Cone15 rows."""
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType


def require(ok,message):
    if not ok:
        raise ValueError(message)


def live_covered(event, wanted, layer, field, pos_key):
    selected=event['selected_positions']
    if pos_key not in event:
        return event[field]==event['selected_rows']
    live=event[pos_key]
    return (21<=layer<=39 and live==sorted(set(live)) and set(wanted)<=set(live)<=set(selected)
            and event[field]==len(live))


def private_tail_proof(base):
    import index20_runtime as source
    cls=next(c for c in base.__mro__ if c.__qualname__=='index20_controller_class.<locals>.Index20Controller')
    file=Path(inspect.getsourcefile(source.mandatory_tail_update_proof))
    require(hashlib.sha256(file.read_bytes()).hexdigest()=='921372bd37734e903f91514f61e7d911fe7060091184a82273fc57087e6bcde1'
            and cls.__dict__['finish'].__globals__ is source.__dict__, 'Expected newly constructed frozen Index20 type')
    src=textwrap.dedent(inspect.getsource(source.mandatory_tail_update_proof))
    for field,pos_key in [('actual_ffn_rows','cone15_ffn_positions'),('actual_attention_rows','cone15_core_positions'),
                          ('actual_residual_rows','cone15_mhc_positions'),('actual_mhc_rows','cone15_mhc_positions')]:
        old='e["'+field+'"]==e["selected_rows"]'
        require(src.count(old)==1,'Unrecognized native row check')
        src=src.replace(old,'physical_rows(e,wanted,layer,'+repr(field)+','+repr(pos_key)+')')
    ns=dict(source.__dict__,physical_rows=live_covered)
    exec(compile(src,'<private Cone15 native tail proof>','exec'),ns)
    orig=cls.__dict__['finish']
    new=FunctionType(orig.__code__,dict(orig.__globals__,mandatory_tail_update_proof=ns['mandatory_tail_update_proof']),
                     orig.__name__,orig.__defaults__,orig.__closure__)
    new.__kwdefaults__=orig.__kwdefaults__
    cls.finish=new
