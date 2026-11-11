"""Request-scoped double-buffer H2D staging. Import is CPU-only.

Only immutable pinned CPU tail banks may enter this object. A slot is not
overwritten until its consumer stream has finished reading it. Restored KV may
be retained by the controller within this phase, never across reset().
"""
import math
from .core import require


class TailStaging:
    def __init__(self, torch_module, banks, *, device, generation):
        self.t, self.device, self.generation = torch_module, device, generation
        require(type(generation) is int and generation > 0, "positive phase generation required")
        self.layers = tuple(sorted(banks))
        require(self.layers == tuple(range(3, 40)), "exact post-three tail staging layers required")
        self.banks, self.records, self.pending, self.acquired = banks, [], {}, set()
        self.closed = False
        first = banks[3]
        require(first.ndim == 2 and first.shape[1] == 512 and first.shape[0] == 1024,
                "eight 128-row tails required")
        for value in banks.values():
            require(value.device.type == "cpu" and value.is_pinned() and value.dtype == self.t.bfloat16
                    and value.shape == first.shape, "immutable pinned CPU tail bank required")
        self.stream = self.t.cuda.Stream(device=device)
        self.buffers = [self.t.empty(first.shape, dtype=self.t.bfloat16, device=device) for _ in range(2)]
        # CUDA's caching allocator associates these allocations with the
        # current stream. A newly created copy stream must not reuse that
        # storage before prior current-stream uses of recycled blocks finish.
        self.stream.wait_stream(self.t.cuda.current_stream(device))
        self.release_events = [None, None]
        self.next_layer = 3
        try:
            self._schedule(3)
            self._schedule(4)
        except BaseException:
            # __init__ has not returned to the controller yet. Complete any
            # first copy before letting its temporary pinned bank go away.
            self.close()
            raise

    def _schedule(self, layer):
        require(not self.closed and layer == self.next_layer and layer <= 39, "nonsequential tail prefetch")
        slot = (layer - 3) % 2
        with self.t.cuda.stream(self.stream):
            if self.release_events[slot] is not None:
                self.stream.wait_event(self.release_events[slot])
            begin, end = self.t.cuda.Event(enable_timing=True), self.t.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            self.buffers[slot].copy_(self.banks[layer], non_blocking=True)
            end.record(self.stream)
        self.pending[layer] = (slot, end)
        self.records.append(dict(layer=layer, slot=slot, generation=self.generation,
                                 bytes=self.banks[layer].numel() * self.banks[layer].element_size(),
                                 begin=begin, end=end))
        self.next_layer += 1

    def acquire(self, layer):
        require(not self.closed and layer in self.pending and layer not in self.acquired,
                "tail prefetch missing or acquired twice")
        slot, done = self.pending[layer]
        self.t.cuda.current_stream(self.device).wait_event(done)
        self.acquired.add(layer)
        return self.buffers[slot]

    def release(self, layer):
        require(not self.closed and layer in self.acquired, "tail staging released before acquire")
        slot, _ = self.pending.pop(layer)
        self.acquired.remove(layer)
        done = self.t.cuda.Event()
        done.record(self.t.cuda.current_stream(self.device))
        self.release_events[slot] = done
        if self.next_layer <= 39:
            self._schedule(self.next_layer)

    def report(self):
        require(not self.acquired and not self.pending and self.next_layer == 40,
                "not every layer's staged bank was consumed")
        # Called after native request completion. These events measure copies,
        # not TTFT; overlapping per-layer copy durations must not be summed as wall.
        copies = []
        for event in self.records:
            require(event["end"].query(), "request ended before tail copy completion")
            elapsed = float(event["begin"].elapsed_time(event["end"]))
            require(math.isfinite(elapsed) and elapsed > 0, "invalid measured H2D event duration")
            copies.append({k: v for k, v in event.items() if k not in ("begin", "end")} |
                          {"cuda_copy_ms": elapsed})
        return dict(schema="tail-link-h2d-staging-v1", phase_generation=self.generation,
                    double_buffers=2, copies=copies, copy_bytes=sum(x["bytes"] for x in copies),
                    copy_count=len(copies), cross_phase_gpu_reuse=False,
                    copies_after_reset=True, same_phase_restored_reuse=True,
                    copy_ms_is_not_wall_or_ttft=True)

    def close(self):
        if not self.closed:
            # Exceptional/early cleanup must not release pinned storage while a
            # nonblocking copy is still reading it. Normal reset charges this
            # operation to the next phase's preparation timing.
            self.stream.synchronize()
            if self.acquired:
                # An exception may occur after acquire and before release.
                # Those reads have no release event yet; finish their owner
                # stream before reclaiming either staging buffer.
                self.t.cuda.current_stream(self.device).synchronize()
            for event in self.release_events:
                if event is not None:
                    event.synchronize()
            self.buffers.clear()
            self.pending.clear()
            self.banks = {}
            self.closed = True
