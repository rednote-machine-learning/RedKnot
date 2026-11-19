"""Owner-local causal SWA dependency plan for the mandatory-tail Cone15 policy.

All returned coordinates are absolute request positions. No Torch/CUDA or model state.
"""
from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json


REFRESH = (20, 24, 28, 32, 36)
D_SPARK = (37, 38, 39)
SET_BITS = tuple(tuple(j for j in range(8) if x & (1 << j)) for x in range(256))
FIELDS = ('inputs','queries','online_kv','offline_kv','offline_project','local_memory',
          'carry_in','carry_out','all_carry_in','all_carry_out','dspark','decode_ring',
          'inherited_index','refresh_index_query')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ordered(values, total, label):
    rows = tuple(values)
    require(all(type(x) is int and 0 <= x < total for x in rows),
            label+' contains a position outside this owner')
    require(all(a < b for a,b in zip(rows,rows[1:])), label+' must be sorted and distinct')
    return rows


def to_mask(rows, n):
    raw = bytearray((n+7)//8)
    for p in rows:
        raw[p//8] |= 1 << (p%8)
    return int.from_bytes(raw,'little')


def span(start, end):
    return ((1 << (end-start))-1) << start


def row_ids(mask, start, end):
    if not mask or end <= start:
        return ()
    local = (mask >> start) & ((1 << (end-start))-1)
    raw = local.to_bytes((local.bit_length()+7)//8,'little')
    return tuple(start+8*i+j for i,byte in enumerate(raw) if byte for j in SET_BITS[byte])


def windows(rows):
    # Seven doublings cover offsets 0 through 127, inclusive.
    for k in (1,2,4,8,16,32,64):
        rows |= rows >> k
    return rows


def row_receipt(rows):
    payload = json.dumps(rows,separators=(',',':'),allow_nan=False).encode()
    return dict(rows=len(rows),sha256=hashlib.sha256(payload).hexdigest())


def local_indices(source, wanted):
    """Absolute-id subsetting; callers can materialize this host map once on their owner GPU."""
    result = tuple(bisect_left(source,p) for p in wanted)
    require(all(i<len(source) and source[i]==p for p,i in zip(wanted,result)),
            'Requested absolute row is absent from source')
    return result


@dataclass(frozen=True)
class LayerMasks:
    inputs:int
    queries:int
    online:int
    offline:int
    index_source:int
    refresh:int


@dataclass(frozen=True)
class ConeGroup:
    plan:object
    layer:int
    start:int
    end:int

    @cached_property
    def state(self):
        return self.plan.layers[self.layer]

    @property
    def index_source(self):
        return self.state.index_source

    @cached_property
    def inputs(self):
        return row_ids(self.state.inputs,self.start,self.end)

    @cached_property
    def queries(self):
        return row_ids(self.state.queries,self.start,self.end)

    @cached_property
    def online_kv(self):
        # Current source positions required by local and/or future groups of this same owner.
        return row_ids(self.state.online,self.start,self.end)

    @cached_property
    def offline_project(self):
        # Current group's offline sources needed anywhere, including following groups.
        return row_ids(self.state.offline,self.start,self.end)

    @cached_property
    def _local_keys(self):
        active = self.state.queries & span(self.start,self.end)
        ring = self.plan.ring if self.end==self.plan.total else 0
        return windows(active) | ring

    @cached_property
    def local_memory(self):
        return row_ids(self._local_keys,max(0,self.start-127),self.end)

    @cached_property
    def offline_kv(self):
        # Absolute rows consumed by this group's active queries or final decode store.
        return row_ids(self._local_keys & self.state.offline,max(0,self.start-127),self.end)

    def _carry(self, boundary):
        # Earlier source rows required by a query at/after boundary. Includes carry-through across empty groups.
        if boundary in (0,self.plan.total):
            return 0
        future = self.state.queries >> boundary
        # A future query q consumes previous keys >= q-127. The earliest q has the widest left reach.
        first = boundary+(future & -future).bit_length()-1 if future else self.plan.total+127
        low = max(0,first-127)
        keep = span(low,boundary) if low<boundary else 0
        return keep | (self.plan.ring & span(0,boundary))

    @cached_property
    def _before(self):
        return self._carry(self.start)

    @cached_property
    def _after(self):
        return self._carry(self.end)

    @cached_property
    def all_carry_in(self):
        return row_ids(self._before,max(0,self.start-127),self.start)

    @cached_property
    def all_carry_out(self):
        return row_ids(self._after,max(0,self.end-127),self.end)

    @cached_property
    def carry_in(self):
        return row_ids(self._before & self.plan.selected_mask,max(0,self.start-127),self.start)

    @cached_property
    def carry_out(self):
        return row_ids(self._after & self.plan.selected_mask,max(0,self.end-127),self.end)

    @cached_property
    def dspark(self):
        if self.layer not in D_SPARK:
            return ()
        return row_ids(self.plan.ring,self.start,self.end)

    @cached_property
    def decode_ring(self):
        # Physical store occurs only in the terminal group. May read carry if its final native frame is tiny.
        return self.plan.decode_positions if self.end==self.plan.total else ()

    @cached_property
    def inherited_index(self):
        return self.queries if self.index_source!=self.layer else ()

    @cached_property
    def refresh_index_query(self):
        return row_ids(self.state.refresh,self.start,self.end)

    def receipt(self):
        result = {key:row_receipt(getattr(self,key)) for key in FIELDS}
        return dict(layer=self.layer,start=self.start,end=self.end,index_source=self.index_source,**result)


class Cone15Plan:
    """One new plan per request and generation, after its fixed layer-2 max selection. CPU only."""
    def __init__(self,total,selected,mandatory,*,last_frame_start,group_ranges=None):
        require(type(total) is int and total>0,'Invalid input length')
        require(type(last_frame_start) is int and 0<=last_frame_start<total,'Invalid final native frame start')
        self.total = total
        self.last_frame_start = last_frame_start
        self.selected = ordered(selected,total,'selected')
        self.mandatory = ordered(mandatory,total,'mandatory')
        self.selected_mask = to_mask(self.selected,total)
        self.mandatory_mask = to_mask(self.mandatory,total)
        self.decode_positions = tuple(range(max(0,total-128),total))
        self.ring = span(self.decode_positions[0],total)
        require(self.mandatory_mask & ~self.selected_mask == 0,'Missing original mandatory positions')
        require(self.ring & ~self.selected_mask == 0,'D-Spark tail missing from online selection')
        self.sampler = self.selected_mask & span(max(total-140,last_frame_start),total)
        self.group_ranges = None
        if group_ranges is not None:
            self.validate_group_ranges(group_ranges)
        self.layers = {}
        index_needs = {l:0 for l in REFRESH}
        next_hidden = self.sampler
        for l in range(39,20,-1):
            query = self.mandatory_mask | next_hidden
            source = max(s for s in REFRESH if s<=l)
            index_needs[source] |= query
            refresh = index_needs.get(l,0)
            swa = windows(query) | self.ring
            online = swa & self.selected_mask
            offline = swa & ~self.selected_mask
            hidden = query | online | refresh | (self.ring if l in D_SPARK else 0)
            require(not refresh & ~query and not hidden & ~self.selected_mask,
                    'Unsafe shared-index or hidden dependency')
            self.layers[l] = LayerMasks(hidden,query,online,offline,source,refresh)
            next_hidden = hidden
        self.source20_queries_mask = index_needs[20]

    def validate_group_ranges(self,group_ranges):
        ranges = tuple(tuple(pair) for pair in group_ranges)
        require(ranges and all(len(pair)==2 and all(type(x)is int for x in pair)
                              and 0<=pair[0]<pair[1]<=self.total for pair in ranges),
                'Invalid native group range')
        require(ranges[0][0]==0 and ranges[-1][1]==self.total and
                all(x[1]==y[0] for x,y in zip(ranges,ranges[1:])) and
                ranges[-1][0]<=self.last_frame_start,'Ranges do not partition this request/final frame')
        require(self.group_ranges is None or self.group_ranges==ranges,'Native group boundaries changed')
        self.group_ranges=ranges
        return ranges

    def _check_group(self,start,end):
        require(type(start)is int and type(end)is int and 0<=start<end<=self.total and
                (self.group_ranges is None or (start,end) in self.group_ranges),'Unknown owner group')

    def selection(self,start,end):
        self._check_group(start,end)
        return self.selected[bisect_left(self.selected,start):bisect_left(self.selected,end)]

    def group(self,layer,start,end):
        self._check_group(start,end)
        require(type(layer) is int and 21<=layer<=39,'Expected post-source20 layer')
        return ConeGroup(self,layer,start,end)

    def receipt(self):
        require(self.group_ranges is not None,'Bind actual group ranges before a complete audit')
        return dict(schema='cone15-mandatory-global20-fixed-swa-v1',total=self.total,
            selected=row_receipt(self.selected),mandatory=row_receipt(self.mandatory),
            source20_queries=row_receipt(row_ids(self.source20_queries_mask,0,self.total)),
            groups=[self.group(l,a,b).receipt() for a,b in self.group_ranges for l in range(21,40)])
