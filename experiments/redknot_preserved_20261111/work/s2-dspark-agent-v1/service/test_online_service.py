import json
import queue
import threading
import time
import unittest
import urllib.request

from engine_bridge import TokenRequest
from online_service import ChatService, LiveWorkerPool, make_http_server
import test_checkpoint_codec as codec_tests


class FakeChatPool:
    ready = {i: {} for i in range(8)}
    failed = None

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []

    def run(self, request, on_event):
        self.requests.append(request)
        ids = self.tokenizer.encode("Checked.</think>Done.<｜end▁of▁sentence｜>", add_special_tokens=False).ids
        for index, token in enumerate(ids):
            on_event(dict(event="raw_token", request_id=request.request_id, token_id=token,
                finished=index == len(ids)-1, finish_reason="stop" if index == len(ids)-1 else ""))
        return dict(request_id=request.request_id, arm=request.arm,
            input_tokens=len(request.prompt_token_ids), output_tokens=len(ids), output_ids=ids)


class OnlineServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        codec_tests.CheckpointCodecTests.setUpClass()

    def setUp(self):
        self.pool = FakeChatPool(codec_tests.CheckpointCodecTests.tokenizer)
        self.records = []
        self.service = ChatService(self.pool, codec_tests.CheckpointCodecTests.codec, arm="full",
            model_name="deepseek-v4.1-flash", record=self.records.append)

    def test_live_chat_protocol_response_uses_same_codec(self):
        events = []
        result = self.service.complete(dict(model="deepseek-v4.1-flash",
            messages=[dict(role="user", content="Inspect code.")], max_tokens=32),
            received_s=time.perf_counter(), send_delta=lambda rid, d: events.append(d))
        self.assertEqual(result["choices"][0]["message"]["content"], "Done.")
        self.assertEqual(result["choices"][0]["message"]["reasoning_content"], "Checked.")
        self.assertEqual(self.pool.requests[0].max_tokens, 32)
        self.assertFalse(self.records[0]["client_first_byte_measured"])

    def test_loopback_http_sse_and_usage(self):
        server = make_http_server(self.service, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            data = json.dumps(dict(model="deepseek-v4.1-flash", stream=True,
                stream_options=dict(include_usage=True), messages=[dict(role="user", content="Inspect.")],
                max_tokens=32)).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                lines = response.read().decode().splitlines()
            self.assertEqual(lines[-2], "data: [DONE]")
            chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
            self.assertEqual(chunks[-1]["choices"], [])
            self.assertIn("usage", chunks[-1])
            self.assertTrue(self.records[0]["server_http_flush_measured"])
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_no_public_binding(self):
        with self.assertRaises(ValueError):
            make_http_server(self.service, host="0.0.0.0")

    def test_pool_broadcasts_to_all_ranks_and_drains_callback_failure(self):
        commands, events = [queue.Queue() for _ in range(8)], queue.Queue()
        pool = LiveWorkerPool(commands, events, workers_alive=lambda: True)
        pool.ready = {rank: {} for rank in range(8)}
        request = TokenRequest("r", (0, 5), 8, "full", time.perf_counter())
        events.put(dict(event="raw_token", rank=0, request_id="r"))
        events.put(dict(event="request_complete", rank=0, report=dict(request_id="r")))
        def disconnect(event):
            raise BrokenPipeError("client closed")
        with self.assertRaises(BrokenPipeError) as raised:
            pool.run(request, disconnect)
        self.assertEqual(raised.exception.native_report, dict(request_id="r"))
        self.assertTrue(events.empty())
        self.assertIsNone(pool.failed)
        self.assertTrue(all(command.get_nowait() == request for command in commands))


if __name__ == "__main__":
    unittest.main()
