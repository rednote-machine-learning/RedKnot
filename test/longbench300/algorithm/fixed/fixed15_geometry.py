"""Exact host geometry for fixed15. Install on one controller for one request."""
from bisect import bisect_left
from types import FunctionType, MethodType


def window_union(positions):
    """Sorted union of the inclusive causal windows of sorted distinct positions."""
    memory = []
    last = -1
    for pos in positions:
        memory.extend(range(max(last + 1, pos - 127), pos + 1))
        last = pos
    return memory


class Geometry:
    def __init__(self, documents, total, selected, plan):
        self.total = total
        self.docs = plan.normalize_documents(documents, total)
        self.selected = tuple(plan.validate_selection(self.docs, total, selected))
        self.online = frozenset(self.selected)
        self.digest = plan.digest
        self.require = plan.require
        self.selection_hash = self.digest(self.selected)
        # 0: new, 1: document interior, 2: mandatory tail.
        self.categories = bytearray(total)
        for doc in self.docs:
            start, end = doc["start"], doc["end"]
            self.categories[start:end - 128] = b"\x01" * (end-start-128)
            self.categories[end-128:end] = b"\x02" * 128
        self.offline = tuple(p for p in window_union(self.selected) if p not in self.online)
        self.offline_rows = {p: i for i, p in enumerate(self.offline)}
        self.key = self.signature(documents, total, self.selected)

    @staticmethod
    def signature(docs, total, selected):
        return total, tuple((d["start"], d["end"], d["key"]) for d in docs), tuple(selected)

    def positions(self, start, end):
        return list(self.selected[bisect_left(self.selected, start):bisect_left(self.selected, end)])

    def make_plan(self, start, end):
        self.require(type(start) is type(end) is int and 0 <= start < end <= self.total,
                     "invalid execution frame")
        positions = self.positions(start, end)
        old_start = max(start, min(end-1, max(0, self.total-140)))
        outputs = list(range(old_start, end))
        first_source = bisect_left(positions, old_start)
        # outputs is contiguous at the end of this frame.
        source = list(range(first_source, len(positions)))
        missing = [p for p in window_union(positions) if p not in self.online]
        value = dict(schema="index20-chunk-plan-v1", chunk_start=start, chunk_end=end,
            full_input_tokens=self.total, full_rows=end-start, selected_positions=positions,
            selected_local_rows=[p-start for p in positions], selected_rows=len(positions),
            tail_positions=[p for p in positions if self.categories[p] == 2],
            nondoc_positions=[p for p in positions if self.categories[p] == 0],
            index_added_positions=[p for p in positions if self.categories[p] == 1],
            required_offline_halo_positions=missing, sampler_output_positions=outputs,
            sampler_destination_rows=[positions[i] - old_start for i in source],
            sampler_source_rows=source, native_local_window=128, extra_cross_document_links=False,
            selected_survive_layer39=True, native_yoco140_query_intersection=False,
            original_absolute_positions=True, unselected_deep_residual_mhc=False,
            offline_swa_source="full_normalized_document_swa_bank",
            selection_sha256=self.selection_hash)
        return dict(value, sha256=self.digest(value))

    def local_memory(self, start, end, positions, previous, offline):
        self.require(list(positions) == self.positions(start, end), "current query positions differ")
        previous = list(previous)
        self.require(previous == sorted(set(previous)) and
                     all(max(0, start-127) <= p < start and p in self.online for p in previous),
                     "invalid previous online positions")
        halo = self.offline_rows
        if offline is not None and tuple(offline) != self.offline:
            # Compatibility for callers supplying a sorted superset of required halo.
            self.require(list(offline) == sorted(set(offline)) and not self.online.intersection(offline),
                         "invalid offline source order or shadow")
            halo = {p: i for i, p in enumerate(offline)}
        current = {p:i for i,p in enumerate(positions)}
        previous_rows = {p:i for i,p in enumerate(previous)}
        memory = window_union(positions)
        kinds = []
        rows = []
        for p in memory:
            if p in current:
                kind, row = "current", current[p]
            elif p in previous_rows:
                kind, row = "previous", previous_rows[p]
            elif p in halo:
                kind, row = "halo", halo[p]
            else:
                raise ValueError("missing native SWA dependency at absolute position "+str(p))
            kinds.append(kind)
            rows.append(row)
        value = dict(chunk_start=start, chunk_end=end, query_positions=list(positions),
            memory_positions=memory, source_kinds=kinds, source_rows=rows,
            native_window_size=128, extra_nonlocal_swa_links=False,
            local_lengths=[min(128,p+1) for p in positions],
            offline_swa_source="full_normalized_document_swa_bank",
            selection_sha256=self.selection_hash, all_dependencies_covered=True)
        return dict(value, sha256=self.digest(value))


class GeometryHooks:
    def __init__(self, plan):
        self.plan = plan
        self.current = None
        self.original_attributes = []
        self.calls = {"chunk": 0, "local": 0, "offline": 0}

    def geometry(self, docs, total, selected):
        if self.current is None or self.current.key != Geometry.signature(docs, total, selected):
            self.current = Geometry(docs, total, selected, self.plan)
        return self.current

    def chunk(self, docs, total, selected, start, end):
        self.calls["chunk"] += 1
        return self.geometry(docs, total, selected).make_plan(start, end)

    def local(self, docs, total, selected, start, end, positions,
              previous_positions=(), offline_positions=None):
        self.calls["local"] += 1
        return self.geometry(docs, total, selected).local_memory(
            start, end, positions, previous_positions, offline_positions)

    def offline(self, docs, total, selected):
        self.calls["offline"] += 1
        return list(self.geometry(docs, total, selected).offline)

    def bind(self, target, method_name, global_name, replacement):
        method = getattr(target, method_name)
        func = method.__func__
        # Only the globals of this individual bound method are specialized.
        if global_name not in func.__code__.co_names or func.__globals__.get(global_name) is not getattr(self.plan, global_name):
            raise ValueError("different geometry call site: "+method_name)
        old = vars(target).get(method_name)
        namespace = dict(func.__globals__, **{global_name: replacement})
        copy = FunctionType(func.__code__, namespace, func.__name__, func.__defaults__, func.__closure__)
        copy.__kwdefaults__ = func.__kwdefaults__
        bound = MethodType(copy, target)
        setattr(target, method_name, bound)
        self.original_attributes.append((target, method_name, old, bound))

    def restore(self):
        for target, name, old, bound in reversed(self.original_attributes):
            if vars(target).get(name) is not bound:
                raise RuntimeError("geometry hook ownership changed: "+name)
            if old is None:
                delattr(target, name)
            else:
                setattr(target, name, old)
        self.original_attributes.clear()
        self.current = None


def configure_geometry(controller, enabled):
    """Restore prior bindings, then optionally install isolated per-request geometry."""
    hooks = getattr(controller, "_fixed15_geometry", None)
    if hooks is not None:
        hooks.restore()
        controller._fixed15_geometry = None
    if enabled:
        import index20_plan
        hooks = GeometryHooks(index20_plan)
        try:
            hooks.bind(controller, "_make_deep_group", "make_plan", hooks.chunk)
            hooks.bind(controller.cache, "bind_selection", "required_offline_positions", hooks.offline)
            hooks.bind(controller.attention, "_candidate_event", "local_memory_plan", hooks.local)
        except Exception:
            hooks.restore()
            raise
        controller._fixed15_geometry = hooks
