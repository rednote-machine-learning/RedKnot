"""Loopback OpenAI chat transport for a live, serial eight-rank engine pool."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .engine_bridge import TokenRequest, run_serial_request
except ImportError:
    from engine_bridge import TokenRequest, run_serial_request


def rank_worker(rank, command_queue, event_queue, initialize):
    """Initialize ONCE; each HTTP command drives the same native engine instance.

    initialize(rank) returns (port, barrier, prepare, finish, close, metadata).
    Qualifications and hooks are provided by the run's pinned runtime adapter.
    A worker exception is fatal to the entire group, reported to its supervisor.
    """
    close = None
    try:
        port, barrier, prepare, finish, close, metadata = initialize(rank)
        event_queue.put(dict(event="ready", rank=rank, metadata=metadata))
        while True:
            request = command_queue.get()
            if request is None:
                break
            run_serial_request(port, request, rank=rank, barrier=barrier,
                prepare=prepare, finish=finish,
                emit=lambda event: event_queue.put(dict(rank=rank, **event)))
    except BaseException as error:
        event_queue.put(dict(event="fatal", rank=rank, error=repr(error)))
        recorder = getattr(initialize, "record_failure", None)
        if recorder is not None:
            recorder(rank, error)
        raise
    finally:
        if close is not None:
            close()


class LiveWorkerPool:
    def __init__(self, command_queues, event_queue, *, workers_alive, timeout_s=1800):
        if len(command_queues) != 8:
            raise ValueError("this service requires the qualified EP8/DP8 layout")
        self.commands, self.events = command_queues, event_queue
        self.workers_alive, self.timeout_s = workers_alive, timeout_s
        self.lock = threading.Lock()
        self.ready, self.failed = {}, None

    def receive(self, deadline):
        while time.perf_counter() < deadline:
            try:
                return self.events.get(timeout=.2)
            except queue.Empty:
                if not self.workers_alive():
                    raise RuntimeError("native worker exited; terminate the worker group")
        raise TimeoutError("live native worker response timeout; terminate the worker group")

    def wait_ready(self, timeout_s=600):
        deadline = time.perf_counter() + timeout_s
        while len(self.ready) < 8:
            event = self.receive(deadline)
            if event.get("event") != "ready" or event.get("rank") in self.ready:
                raise RuntimeError("worker initialization failed: " + repr(event))
            self.ready[event["rank"]] = event["metadata"]
        return self.ready

    def run(self, request, on_event):
        # Admission is serial, and waiting remains inside received_s -> TTFT.
        with self.lock:
            if self.failed or len(self.ready) != 8:
                raise RuntimeError("native pool is not ready: " + str(self.failed))
            for commands in self.commands:
                commands.put(request)
            deadline = time.perf_counter() + self.timeout_s
            callback_error = None
            try:
                while True:
                    event = self.receive(deadline)
                    if event.get("event") == "fatal":
                        raise RuntimeError("native worker failed: " + repr(event))
                    if event.get("rank") != 0 or event.get("event") not in ("raw_token", "request_complete"):
                        raise RuntimeError("unexpected live worker event: " + repr(event))
                    rid = event.get("request_id", event.get("report", {}).get("request_id"))
                    if rid != request.request_id:
                        raise RuntimeError("cross-request live stream event")
                    if callback_error is None:
                        try:
                            on_event(event)
                        except BaseException as error:
                            # Drain the actual request even if HTTP disconnects
                            # or parsing fails. Never release the serial slot early.
                            callback_error = error
                    if event["event"] == "request_complete":
                        if callback_error:
                            callback_error.native_report = event["report"]
                            raise callback_error
                        return event["report"]
            except BaseException as error:
                # A drained callback failure is safe for the engine; a transport
                # or rank failure is not. Preserve explicit group-fatal state.
                if error is not callback_error:
                    self.failed = repr(error)
                raise


class ChatService:
    def __init__(self, pool, codec, *, arm, model_name, record):
        if arm not in ("full", "s2", "prefix"):
            raise ValueError("explicit service arm required")
        self.pool, self.codec, self.arm, self.model_name, self.record = pool, codec, arm, model_name, record

    def complete(self, payload, *, received_s, send_delta):
        if payload.get("model") != self.model_name:
            raise ValueError("unknown model")
        if payload.get("n", 1) != 1 or payload.get("logprobs", False):
            raise ValueError("only one completion without logprob export supported")
        for key in ("presence_penalty", "frequency_penalty"):
            if payload.get(key, 0) not in (None, 0):
                raise ValueError(key+" is not implemented by the pinned native sampler")
        if payload.get("logit_bias") not in (None, {}) or payload.get("top_k") not in (None, -1):
            raise ValueError("custom logit_bias/top_k not supported")
        rendered = self.codec.render(payload)
        rendered_s = time.perf_counter()
        rid = uuid.uuid4().hex
        max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if max_tokens is None:
            max_tokens = 270336-len(rendered["input_ids"])
        request = TokenRequest(rid, rendered["input_ids"], max_tokens, self.arm, received_s,
            temperature=payload.get("temperature", 0.0), top_p=payload.get("top_p", 1.0),
            seed=payload.get("seed", 0), documents=tuple(rendered["documents"]))
        request.validate()
        stream = self.codec.completion_stream(rid)
        message = dict(role="assistant", content="", reasoning_content="", tool_calls=[])
        transport = dict(first_delta_s=None, first_content_delta_s=None,
                         first_http_flush_s=None, finish_reason=None)

        def consume(event):
            if event["event"] != "raw_token":
                return
            for delta in stream.feed(event["token_id"], finished=event["finished"],
                                     finish_reason=event["finish_reason"]):
                if "finish_reason" in delta:
                    transport["finish_reason"] = delta["finish_reason"]
                else:
                    for key in ("content", "reasoning_content"):
                        if key in delta:
                            message[key] += delta[key]
                    for call in delta.get("tool_calls", []):
                        message["tool_calls"].append({k: v for k, v in call.items() if k != "index"})
                    now = time.perf_counter()
                    if transport["first_delta_s"] is None:
                        transport["first_delta_s"] = now
                    if delta.get("content") and transport["first_content_delta_s"] is None:
                        transport["first_content_delta_s"] = now
                flushed = send_delta(rid, delta)
                if flushed is not None and transport["first_http_flush_s"] is None:
                    transport["first_http_flush_s"] = flushed

        try:
            report = self.pool.run(request, consume)
        except BaseException as error:
            self.record(dict(schema="s2-live-chat-failure-v1", request_id=rid, arm=self.arm,
                received_s=received_s, failed_s=time.perf_counter(), error=repr(error),
                native_report=getattr(error, "native_report", None), **transport))
            raise
        if not stream.closed:
            raise RuntimeError("native completion did not close chat protocol")
        report.update(transport, encoder_sha256=rendered["encoder_sha256"],
            render_s=rendered_s-received_s, generation_token_budget=max_tokens,
            generation_budget_source="explicit_client_cap" if ("max_tokens" in payload or "max_completion_tokens" in payload)
                else "remaining_native_context_capacity",
            document_spans=rendered["documents"], client_first_byte_measured=False,
            server_http_flush_measured=transport["first_http_flush_s"] is not None)
        self.record(report)
        response = dict(id="chatcmpl-"+rid, object="chat.completion", created=int(time.time()),
            model=self.model_name, choices=[dict(index=0, message=message,
                finish_reason=transport["finish_reason"])], usage=dict(prompt_tokens=report["input_tokens"],
                completion_tokens=report["output_tokens"], total_tokens=report["input_tokens"]+report["output_tokens"]))
        return response


def make_http_server(service, *, host="127.0.0.1", port=8123):
    if host != "127.0.0.1":
        raise ValueError("benchmark service must bind loopback only")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def json_response(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self.json_response(200 if len(service.pool.ready) == 8 and not service.pool.failed else 503,
                    dict(ready=len(service.pool.ready) == 8, arm=service.arm,
                         concurrency=1, failure=service.pool.failed))
            elif self.path == "/v1/models":
                self.json_response(200, dict(object="list", data=[dict(id=service.model_name, object="model")]))
            else:
                self.json_response(404, dict(error="unknown route"))

        def do_POST(self):
            received = time.perf_counter()
            started = False
            if self.path != "/v1/chat/completions":
                self.json_response(404, dict(error="unknown route")); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32*1024**2:
                    raise ValueError("request body length missing or excessive")
                body = json.loads(self.rfile.read(length))
                streaming = bool(body.get("stream", False))

                def delta(rid, value):
                    nonlocal started
                    if not streaming:
                        return
                    if not started:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                    finish = value.get("finish_reason")
                    chunk = dict(id="chatcmpl-"+rid, object="chat.completion.chunk", created=int(time.time()),
                        model=service.model_name, choices=[dict(index=0, delta={} if finish else value, finish_reason=finish)])
                    self.wfile.write(("data: "+json.dumps(chunk, ensure_ascii=False)+"\n\n").encode())
                    self.wfile.flush()
                    return time.perf_counter()

                result = service.complete(body, received_s=received, send_delta=delta)
                if streaming:
                    if body.get("stream_options", {}).get("include_usage"):
                        self.wfile.write(("data: "+json.dumps(dict(id=result["id"], object="chat.completion.chunk",
                            created=result["created"], model=service.model_name, choices=[], usage=result["usage"]))+"\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    self.close_connection = True
                else:
                    self.json_response(200, result)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as error:
                if not started:
                    self.json_response(400 if isinstance(error, ValueError) else 500,
                        dict(error=dict(message=str(error), type=type(error).__name__)))
                else:
                    self.wfile.write(("data: "+json.dumps(dict(error=str(error)))+"\n\n").encode())
                    self.wfile.flush()
                    self.close_connection = True

    return ThreadingHTTPServer((host, port), Handler)
