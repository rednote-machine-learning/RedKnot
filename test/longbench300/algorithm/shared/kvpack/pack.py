"""Bounded, owner-scoped cross-layer KV gather and memory scatter.

Each lease belongs to one immutable owner/input/bank epoch and stage/group.
Returned views are borrowed until release and must be consumed on the creating
CUDA stream. Selected online hidden/projection execution remains layer ordered.
"""
from dataclasses import dataclass, field
import hashlib
import json
from dataclasses import replace


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_stamp(value):
    """Use available versions, explicitly retaining no-version inference identity.

    Native prepared halos and original cached indices are immutable by their
    owning runtime contract, but inference tensors do not expose a version
    counter. Do not invent one. Own32's independent byte oracle remains needed.
    """
    try:version=value._version
    except RuntimeError:
        require(callable(getattr(value,'is_inference',None))and value.is_inference(),
                'tensor version unavailable outside native inference mode')
        version=None
    return (id(value),value.data_ptr(),version,tuple(value.shape),
            tuple(value.stride())if callable(getattr(value,'stride',None))else None,
            str(value.dtype),str(value.device))


@dataclass(frozen=True)
class Document:
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class PackPlan:
    layers: tuple
    documents: tuple
    positions: tuple
    document_indices: tuple
    local_rows: tuple
    owner_key: str
    stage_key: str
    group_key: str
    cache_binding: object = field(default=None,repr=False,compare=False)

    @classmethod
    def build(cls, layers, documents, positions, *, owner_key, stage_key, group_key):
        layers, documents, positions = tuple(layers), tuple(documents), tuple(positions)
        require(layers and len(layers) == len(set(layers)) and all(type(x) is int and 3 <= x < 39 for x in layers),
                'offline full-SWA pack supports unique layers3..38; regenerated layer39 stays native')
        require(all(isinstance(x, str) and x for x in (owner_key, stage_key, group_key)), 'explicit owner/stage/group required')
        require(positions == tuple(sorted(set(positions))) and all(type(p) is int and p >= 0 for p in positions),
                'absolute offline positions must be sorted unique nonnegative integers')
        end = 0
        for doc in documents:
            require(isinstance(doc, Document) and isinstance(doc.key, str) and doc.key
                    and type(doc.start) is int and type(doc.end) is int
                    and end <= doc.start < doc.end, 'documents must have complete ordered nonoverlapping ranges')
            end = doc.end
        indices, rows = [], []
        for p in positions:
            hits = [i for i, doc in enumerate(documents) if doc.start <= p < doc.end]
            require(len(hits) == 1, 'offline position has no unique source document: ' + str(p))
            i = hits[0]
            indices.append(i)
            rows.append(p - documents[i].start)
        return cls(layers, documents, positions, tuple(indices), tuple(rows), owner_key, stage_key, group_key)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps({k:v for k,v in self.__dict__.items()if k!='cache_binding'}, default=lambda x: x.__dict__,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def tiles(self, max_rows):
        require(type(max_rows) is int and max_rows > 0, 'positive row tile required')
        return [PackPlan.build(self.layers, self.documents, self.positions[a:a+max_rows],
            owner_key=self.owner_key, stage_key=self.stage_key,
            group_key=self.group_key + '/rows' + str(a)) for a in range(0, len(self.positions), max_rows)]

    def for_layers(self,layers):
        layers=tuple(layers)
        require(layers and len(layers)==len(set(layers))and set(layers)<=set(self.layers),'window layers outside admitted stage')
        return replace(self,layers=layers)


def gather_plan_from_cache(cache,layers,*,owner_key,stage_key,group_key):
    """O(Noffline + documents) mapping from the actual original capture groups.

    Call once per selection stage, then plan.for_layers(window). Repeated
    document keys are resolved only against their actual absolute occurrence.
    """
    layers=tuple(layers)
    require(layers and len(layers)==len(set(layers))and all(type(l)is int and 3<=l<39 for l in layers),
            'admitted offline stage layers3..38 required')
    require(all(isinstance(x,str)and x for x in(owner_key,stage_key,group_key)),'owner/stage/group required')
    documents=tuple(Document(d['key'],d['start'],d['end'])for d in cache.docs)
    prior=0;by_key={}
    for i,d in enumerate(documents):
        require(type(d.start)is int and type(d.end)is int and prior<=d.start<d.end,'invalid cache document namespace')
        prior=d.end;by_key.setdefault(d.key,[]).append(i)
    positions=tuple(cache.index20_offline_positions)
    require(all(type(p)is int and p>=0 for p in positions)
            and all(a<b for a,b in zip(positions,positions[1:])),'offline absolute positions must be increasing')
    ids=[-1]*len(positions);rows=[-1]*len(positions)
    for key,dst,src in cache.index20_halo_groups:
        require(key in by_key and len(dst)==len(src),'source document/local-row group differs')
        for destination,local in zip(dst,src):
            require(type(destination)is int and 0<=destination<len(positions)and ids[destination]==-1
                    and type(local)is int and local>=0,'duplicate/out-of-range original halo mapping')
            choices=[i for i in by_key[key]if documents[i].start+local==positions[destination]
                     and local<documents[i].end-documents[i].start]
            require(len(choices)==1,'original bank row is not the declared absolute input position')
            ids[destination],rows[destination]=choices[0],local
    require(all(i>=0 for i in ids),'original halo groups omit an offline position')
    return PackPlan(layers,documents,positions,tuple(ids),tuple(rows),owner_key,stage_key,group_key,
                    (cache,cache.index20_offline_positions,cache.index20_halo_groups))


@dataclass(frozen=True)
class CachedHaloGeometry:
    """One validated original mapping; borrow its immutable cached GPU indices.

    Original _mapping/_indices owns CPU→GPU value binding. This object retains
    those exact tensors, checks their storage/version, and makes no new claim of
    independent GPU readback. Qualification's real KV byte oracle supplies that.
    """
    owner_key: str
    stage_key: str
    group_key: str
    memory_positions: tuple
    source_positions: tuple
    source_rows: tuple
    destination_rows: tuple
    source_rows_gpu: object
    destination_rows_gpu: object
    index_versions: tuple
    mapping_identity: int


def oracle_gather(plan, banks):
    """CPU oracle accepts arbitrary immutable row payloads, including raw BF16 bytes."""
    return tuple(tuple(banks[layer, plan.documents[d].key][r]
                       for d, r in zip(plan.document_indices, plan.local_rows)) for layer in plan.layers)


def oracle_scatter(prepared_layers, destination_rows, memory_rows, *, sentinel=None):
    destination_rows = tuple(destination_rows)
    require(len(destination_rows) == len(set(destination_rows))
            and all(type(r) is int and 0 <= r < memory_rows for r in destination_rows), 'invalid scatter destinations')
    result = []
    for values in prepared_layers:
        require(len(values) == len(destination_rows), 'scatter source length differs')
        row = [sentinel] * memory_rows
        for dst, value in zip(destination_rows, values):
            row[dst] = value
        result.append(tuple(row))
    return tuple(result)


def halo_mapping(memory_positions, source_positions, source_rows, destination_rows):
    memory, positions, src, dst = map(tuple, (memory_positions, source_positions, source_rows, destination_rows))
    require(memory == tuple(sorted(set(memory))) and positions == tuple(sorted(set(positions)))
            and all(type(p) is int and p >= 0 for p in memory + positions), 'invalid absolute position namespace')
    require(len(src) == len(dst) and len(dst) == len(set(dst))
            and all(type(i) is int and 0 <= i < len(positions) for i in src)
            and all(type(i) is int and 0 <= i < len(memory) for i in dst), 'invalid halo source/destination row namespace')
    require(tuple(positions[i] for i in src) == tuple(memory[i] for i in dst),
            'halo source absolute positions do not match scatter destinations')
    return memory, positions, src, dst


def oracle_halo_scatter(layers, halos, source_rows, destination_rows, memory_positions, source_positions, *, sentinel=None):
    memory, positions, src, dst = halo_mapping(memory_positions, source_positions, source_rows, destination_rows)
    require(all(len(halos[layer]) == len(positions) for layer in layers), 'halo source row count differs')
    return oracle_scatter([[halos[layer][i] for i in src] for layer in layers], dst, len(memory), sentinel=sentinel)


class KVPackSession:
    def __init__(self, owner_key, *, max_bytes=256 * 1024**2, torch_module=None):
        require(isinstance(owner_key, str) and owner_key, 'owner/input/bank epoch key required')
        require(type(max_bytes) is int and max_bytes > 0, 'positive bounded byte budget required')
        self.owner_key, self.max_bytes, self.t = owner_key, max_bytes, torch_module
        self.live_bytes, self.high_water_bytes, self.leases = 0, 0, set()
        self.gather_geometry_cache={}

    def reserve(self, size):
        require(self.live_bytes + size <= self.max_bytes, 'KV pack memory budget exceeded; tile rows or layers')
        self.live_bytes += size
        self.high_water_bytes = max(self.high_water_bytes, self.live_bytes)

    def gather(self, plan, banks):
        require(plan.owner_key == self.owner_key, 'owner/input/bank epoch mismatch')
        if self.t is None:
            import torch
            self.t = torch
        return PackLease(self, plan, banks)

    def gather_cache(self, cache, plan):
        """Read each document's original normalized full-SWA bank, without concatenation."""
        if plan.cache_binding is not None:
            require(all(a is b for a,b in zip(plan.cache_binding,
                (cache,cache.index20_offline_positions,cache.index20_halo_groups))),
                'cache owner/selection stage changed after gather plan admission')
        require(tuple(Document(d['key'], d['start'], d['end']) for d in cache.docs) == plan.documents,
                'cache document/absolute-position namespace differs')
        return self.gather(plan, {(layer, doc.key): cache.entry(doc.key, 'swa_full', layer)
                                 for layer in plan.layers for doc in plan.documents})

    def _gather_geometry(self,plan,device):
        key=(plan.stage_key,id(plan.documents),id(plan.document_indices),id(plan.local_rows),str(device))
        if key in self.gather_geometry_cache:
            item=self.gather_geometry_cache[key]
            require(tuple(tensor_stamp(x)for x in item['tensors'])==item['versions'],
                    'cached document/local-row GPU geometry mutated')
            return item,True
        size=2*len(plan.positions)*8
        self.reserve(size)
        try:
            tensors=(self.t.tensor(plan.document_indices,device=device,dtype=self.t.int64),
                     self.t.tensor(plan.local_rows,device=device,dtype=self.t.int64))
            item=dict(tensors=tensors,versions=tuple(tensor_stamp(x)for x in tensors),
                bytes=size,plan=plan)  # Retain CPU tuple identities, preventing ID reuse.
            self.gather_geometry_cache[key]=item
            return item,False
        except BaseException:
            self.live_bytes-=size
            raise

    def release_gather_geometry(self,stage_key):
        require(not any(x.plan.stage_key==stage_key for x in self.leases),'active gather lease still uses this stage')
        for key,item in list(self.gather_geometry_cache.items()):
            if key[0]==stage_key:
                self.live_bytes-=item['bytes'];del self.gather_geometry_cache[key]

    def scatter_halos(self, layers, prepared_halos, source_rows, destination_rows, memory_positions,
                      *, source_positions, stage_key, group_key, account_retained_halos=False):
        """Direct integration: one real data kernel across original prepared halos.

        Existing native _halo/prepare_swa runs unchanged before this call. New
        pack bytes are bounded; optional accounting also includes retained halo
        sources. Their bytes are always reported separately for caller budgets.
        """
        if self.t is None:
            import torch
            self.t = torch
        return PackLease.from_halos(self, layers, prepared_halos, source_rows, destination_rows,
            memory_positions, source_positions=source_positions, stage_key=stage_key,
            group_key=group_key, account_retained_halos=account_retained_halos)

    def geometry_from_mapping(self, mapping, *, source_positions, stage_key, group_key):
        """Call once per stage/group; no CUDA allocation or new index transfer."""
        require(isinstance(stage_key,str)and stage_key and isinstance(group_key,str)and group_key,
                'cached mapping requires stage/group identity')
        event=mapping['event'];kinds=event['source_kinds'];rows=event['source_rows']
        require(len(kinds)==len(rows)==len(event['memory_positions'])
                and all(k in('current','previous','halo')for k in kinds),'uncovered/unknown original memory source')
        dst=tuple(i for i,k in enumerate(kinds)if k=='halo');src=tuple(rows[i]for i in dst)
        require(dst,'cached halo fusion requires actual offline rows')
        memory,positions,src,dst=halo_mapping(event['memory_positions'],source_positions,src,dst)
        transfers=[(d,s)for kind,d,s in mapping['transfers']if kind=='halo']
        require(len(transfers)==1,'one original cached halo transfer required')
        di,si=transfers[0]
        if self.t is None:
            import torch
            self.t=torch
        require(si.is_cuda and di.is_cuda and si.device==di.device and si.ndim==di.ndim==1
                and si.dtype in(self.t.int32,self.t.int64)and di.dtype in(self.t.int32,self.t.int64)
                and tuple(si.shape)==tuple(di.shape)==(len(src),)and si.is_contiguous()and di.is_contiguous(),
                'cached original GPU index tensor shape/dtype/device differs')
        versions=tuple(tensor_stamp(x)for x in(si,di))
        return CachedHaloGeometry(self.owner_key,stage_key,group_key,memory,positions,src,dst,si,di,versions,id(mapping))

    def scatter_cached_halos(self,layers,prepared_halos,geometry,*,account_retained_halos=False):
        require(isinstance(geometry,CachedHaloGeometry)and geometry.owner_key==self.owner_key,
                'cached halo geometry belongs to another owner')
        return PackLease.from_halos(self,layers,prepared_halos,None,None,None,source_positions=None,
            stage_key=geometry.stage_key,group_key=geometry.group_key,
            account_retained_halos=account_retained_halos,cached_geometry=geometry)

    def close(self):
        for lease in tuple(self.leases):
            lease.release()
        for item in self.gather_geometry_cache.values():self.live_bytes-=item['bytes']
        self.gather_geometry_cache.clear()


class PackLease:
    @classmethod
    def from_halos(cls, session, layers, halos, source_rows, destination_rows, memory_positions,
                   *, source_positions, stage_key, group_key, account_retained_halos,cached_geometry=None):
        layers = tuple(layers)
        require(layers and len(set(layers)) == len(layers)
                and all(type(l) is int and 3 <= l < 39 for l in layers), 'unique layers3..38 required; layer39 stays native')
        require(isinstance(stage_key, str) and stage_key and isinstance(group_key, str) and group_key,
                'stage/group binding required')
        if cached_geometry is None:
            memory,positions,src,dst=halo_mapping(memory_positions,source_positions,source_rows,destination_rows)
        else:
            memory,positions,src,dst=(cached_geometry.memory_positions,cached_geometry.source_positions,
                cached_geometry.source_rows,cached_geometry.destination_rows)
        self = object.__new__(cls)
        self.session, self.closed = session, False
        self.plan = PackPlan(layers, (), tuple(memory[i] for i in dst), (), src,
                             session.owner_key, stage_key, group_key)
        self.buffers, self.sources, self.prepared_sources, self.accounted = [], [halos[l] for l in layers], [], 0
        self.staging = self.memory = None
        self.gather_launches = self.scatter_launches = 0
        self.cached_geometry=cached_geometry
        t = session.t
        first = self.sources[0]
        require(first.is_cuda and first.dtype in (t.bfloat16, t.float16, t.float32), 'CUDA native floating halos required')
        self.device, self.dtype = first.device, first.dtype
        require(t.cuda.current_device()==self.device.index,'active CUDA device differs from owned halo bank')
        self.stream = t.cuda.current_stream(self.device)
        require(all(x.device == self.device and x.dtype == self.dtype and x.is_contiguous()
                    and tuple(x.shape) == (len(positions), 1, 512) for x in self.sources),
                'each layer must supply its own contiguous[Noffline,1,512] halo with identical dtype/device')
        self.source_versions = [tensor_stamp(x) for x in self.sources]
        require(not len(positions) or len({x.data_ptr() for x in self.sources}) == len(layers),
                'different layers cannot alias the same prepared halo')
        self.retained_source_bytes = sum(x.numel()*x.element_size() for x in self.sources)
        size = len(layers)*len(memory)*512*first.element_size() + (len(layers)+(0 if cached_geometry else 2*len(src)))*8
        self._reserve(size + (self.retained_source_bytes if account_retained_halos else 0))
        try:
            pointers = t.tensor([x.data_ptr() for x in self.sources], device=self.device, dtype=t.int64)
            if cached_geometry is None:
                sources=t.tensor(src,device=self.device,dtype=t.int64)
                destinations=t.tensor(dst,device=self.device,dtype=t.int64)
            else:
                sources,destinations=cached_geometry.source_rows_gpu,cached_geometry.destination_rows_gpu
                require(sources.device==destinations.device==self.device,'cached indices are on another owner device')
                require(tuple(tensor_stamp(x)for x in(sources,destinations))==cached_geometry.index_versions,
                        'cached mapping indices mutated/rebound')
            self.buffers.extend((pointers, sources, destinations))
            self.memory = t.empty((len(layers), len(memory), 1, 512), device=self.device, dtype=self.dtype)
            self._record(self.sources + self.buffers + [self.memory])
            from .kernels import launch_scatter
            launch_scatter(pointers, sources, destinations, self.memory, len(src))
            self.scatter_launches = int(bool(src))
            self.memory_positions, self.offline_destinations = memory, dst
            self.assembled = set()
            session.leases.add(self)
            return self
        except BaseException:
            self.release()
            raise

    def __init__(self, session, plan, banks):
        self.session, self.plan, self.closed = session, plan, False
        self.staging = self.memory = None
        self.buffers, self.sources, self.prepared_sources, self.accounted = [], [], [], 0
        self.gather_launches = self.scatter_launches = 0
        t = session.t
        require(plan.documents, 'a bank-backed pack requires source documents')
        self.sources = [banks[layer, doc.key] for layer in plan.layers for doc in plan.documents]
        first = self.sources[0]
        require(first.is_cuda and first.dtype == t.bfloat16, 'native normalized SWA banks must be CUDA BF16')
        self.device, self.dtype = first.device, first.dtype
        require(t.cuda.current_device()==self.device.index,'active CUDA device differs from owned normalized bank')
        self.stream = t.cuda.current_stream(self.device)
        self.source_versions = []
        for (layer, doc), source in zip(((l, d) for l in plan.layers for d in plan.documents), self.sources):
            require(source.device == self.device and source.dtype == self.dtype
                    and tuple(source.shape) == (doc.end-doc.start, 512) and source.is_contiguous(),
                    'bank layer/document shape, dtype, row layout or device differs')
            self.source_versions.append(tensor_stamp(source))
        count, layers, docs = len(plan.positions), len(plan.layers), len(plan.documents)
        # Account pointer/index metadata as well as the actual moved row payload.
        geometry,hit=session._gather_geometry(plan,self.device)
        self.gather_geometry,self.gather_geometry_cache_hit=geometry,hit
        self._reserve(layers * count * 512 * 2 + layers * docs * 8)
        try:
            pointers = t.tensor([x.data_ptr() for x in self.sources], device=self.device, dtype=t.int64)
            doc_ids,local_rows=geometry['tensors']
            self.buffers.extend((pointers, doc_ids, local_rows))
            self.staging = t.empty((layers, count, 512), device=self.device, dtype=self.dtype)
            self._record(self.sources + self.buffers + [self.staging])
            from .kernels import launch_gather
            launch_gather(pointers, doc_ids, local_rows, self.staging)
            self.gather_launches = int(count > 0)
            session.leases.add(self)
        except BaseException:
            self.release()
            raise

    def _reserve(self, size):
        self.session.reserve(size)
        self.accounted += size

    def _record(self, tensors):
        for value in tensors:
            value.record_stream(self.stream)

    def _check(self, *, owner_key=None, stage_key=None, group_key=None):
        require(not self.closed, 'released KV pack cannot be consumed')
        p = self.plan
        require((p.owner_key if owner_key is None else owner_key,
                 p.stage_key if stage_key is None else stage_key,
                 p.group_key if group_key is None else group_key)
                == (p.owner_key, p.stage_key, p.group_key), 'owner/stage/group binding differs')
        require(self.session.t.cuda.current_stream(self.device).cuda_stream == self.stream.cuda_stream,
                'KV pack must be consumed on its creating stream')
        require(all(tensor_stamp(x) == old for x, old in zip(self.sources, self.source_versions)),
                'original offline bank mutated/rebound during KV lease')
        geometry=getattr(self,'cached_geometry',None)
        if geometry is not None:
            require(tuple(tensor_stamp(x)for x in(geometry.source_rows_gpu,geometry.destination_rows_gpu))
                    ==geometry.index_versions,'cached mapping indices mutated/rebound')
        gather_geometry=getattr(self,'gather_geometry',None)
        if gather_geometry is not None:
            require(tuple(tensor_stamp(x)for x in gather_geometry['tensors'])==gather_geometry['versions'],
                    'cached gather document/local-row indices mutated/rebound')

    def normalized(self, layer, **binding):
        self._check(**binding)
        require(self.staging is not None, 'prepared-halo scatter lease has no normalized-row staging')
        return self.staging[self.plan.layers.index(layer)]

    def scatter_prepared(self, prepared_by_layer, memory_positions, offline_destinations, **binding):
        """Fused across-layer scatter AFTER original per-layer native prepare_swa.

        Only offline slots are initialized here. assemble_memory fills all online
        slots using this layer's original current/previous values before return.
        """
        self._check(**binding)
        require(self.memory is None, 'prepared memory can only be scattered once')
        t, p = self.session.t, self.plan
        positions, dst = tuple(memory_positions), tuple(offline_destinations)
        require(positions == tuple(sorted(set(positions))) and len(dst) == len(p.positions)
                and len(dst) == len(set(dst)) and all(type(x) is int and 0 <= x < len(positions) for x in dst),
                'memory/scatter row namespace differs')
        require(tuple(positions[x] for x in dst) == p.positions, 'offline absolute positions differ from destination memory rows')
        values = [prepared_by_layer[layer] for layer in p.layers]
        for value in values:
            require(value.device == self.device and value.dtype == self.dtype
                    and tuple(value.shape) == (len(dst), 1, 512) and value.is_contiguous(),
                    'prepared native SWA shape/layout/dtype/device differs')
        # Include externally created prepared values while this lease retains them.
        size = len(p.layers)*len(positions)*512*2 + sum(x.numel()*x.element_size() for x in values)
        size += (len(values) + 2*len(dst))*8
        self._reserve(size)
        try:
            self.prepared_sources = values
            pointers = t.tensor([x.data_ptr() for x in values], device=self.device, dtype=t.int64)
            sources = t.arange(len(dst), device=self.device, dtype=t.int64)
            destinations = t.tensor(dst, device=self.device, dtype=t.int64)
            self.buffers.extend((pointers, sources, destinations))
            self.memory = t.empty((len(p.layers), len(positions), 1, 512), device=self.device, dtype=self.dtype)
            self._record(values + [pointers, sources, destinations, self.memory])
            from .kernels import launch_scatter
            launch_scatter(pointers, sources, destinations, self.memory, len(dst))
            self.scatter_launches = int(bool(dst))
            self.memory_positions, self.offline_destinations = positions, dst
            self.assembled = set()
        except BaseException:
            self.release()
            raise

    def assemble_memory(self, layer, current, previous, *, current_dst, current_src,
                        previous_dst, previous_src, **binding):
        """Layer-ordered online fill; never computes another layer's hidden state."""
        self._check(**binding)
        require(self.memory is not None and layer not in self.assembled, 'no staged memory or layer already consumed')
        t = self.session.t
        destinations = list(self.offline_destinations)
        pairs = [(current, tuple(current_dst), tuple(current_src)), (previous, tuple(previous_dst), tuple(previous_src))]
        for source, dst, src in pairs:
            require(source.device == self.device and source.dtype == self.dtype
                    and source.ndim == 3 and tuple(source.shape[1:]) == (1, 512), 'online SWA source differs')
            require(len(dst) == len(src) and all(type(x) is int and 0 <= x < len(self.memory_positions) for x in dst)
                    and all(type(x) is int and 0 <= x < source.shape[0] for x in src), 'online source/destination bounds differ')
            destinations.extend(dst)
        require(sorted(destinations) == list(range(len(self.memory_positions))), 'memory contains missing/duplicate/uninitialized rows')
        result = self.memory[self.plan.layers.index(layer)]
        for source, dst, src in pairs:
            if dst:
                di = t.tensor(dst, device=self.device, dtype=t.int64)
                si = t.tensor(src, device=self.device, dtype=t.int64)
                self._record([source, di, si])
                result.index_copy_(0, di, source.index_select(0, si))
        self.assembled.add(layer)
        return result

    def report(self):
        return dict(schema='cross-layer-offline-kv-pack-v1', owner_key=self.plan.owner_key,
            stage_key=self.plan.stage_key, group_key=self.plan.group_key, plan_sha256=self.plan.signature,
            layers=list(self.plan.layers), offline_rows=len(self.plan.positions), width=512,
            dtype=str(self.dtype), gather_data_kernel_launches=self.gather_launches,
            scatter_data_kernel_launches=self.scatter_launches, reserved_new_bytes=self.accounted,
            session_high_water_bytes=self.session.high_water_bytes,
            retained_source_halo_bytes=getattr(self,'retained_source_bytes',0),
            original_cached_gpu_indices_reused=getattr(self,'cached_geometry',None)is not None,
            gather_geometry_cache_hit=getattr(self,'gather_geometry_cache_hit',False),
            new_index_h2d_transfers=0 if(getattr(self,'cached_geometry',None)is not None
                or getattr(self,'gather_geometry_cache_hit',False))else 2,
            source_pointer_count=len(self.sources),
            source_inference_tensors_without_version=sum(stamp[2]is None for stamp in self.source_versions),
            source_identity_scope='object/pointer/layout/dtype/device plus version when available; original inference sources remain read-only by runtime contract',
            memory_shape=None if self.memory is None else list(self.memory.shape),
            actual_data_movement='cross_layer_pointer_table_gather_scatter',
            gpu_bitwise_oracle_checked=getattr(self,'oracle_checked',False),
            pointer_table_no_whole_bank_cat=True, original_native_prepare_swa_required=True,
            selected_hidden_layer_order_unchanged=True, closed=self.closed)

    def verify_offline(self):
        """Optional real GPU-byte check; synchronizes, so use diagnostic windows."""
        self._check()
        require(self.memory is not None, 'offline scatter output required for diagnostic')
        t = self.session.t
        dst = t.tensor(self.offline_destinations, device=self.device, dtype=t.int64)
        rows = t.arange(len(self.plan.positions),device=self.device,dtype=t.int64) if self.prepared_sources else t.tensor(
            self.plan.local_rows,device=self.device,dtype=t.int64)
        sources = self.prepared_sources or self.sources
        for i, source in enumerate(sources):
            actual = self.memory[i].index_select(0,dst).contiguous().view(t.uint8)
            expected = source.index_select(0,rows).contiguous().view(t.uint8)
            require(t.equal(actual,expected),'fused offline KV bytes differ from original layer halo')
        self.oracle_checked = True
        return self.report()

    def release(self):
        if self.closed:
            return
        self.closed = True
        self.session.live_bytes -= self.accounted
        self.session.leases.discard(self)
        self.accounted = 0
        self.staging = self.memory = None
        self.buffers.clear()
        self.sources.clear()
        self.prepared_sources.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
