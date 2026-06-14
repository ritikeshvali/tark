# design decisions

records of architectural decisions made while building tark: what we chose, what we 
rejected, and why. updated as new features are added.

---

## HTTP server + single request inference

### library choices

**HTTP server: cpp-httplib**
- single header, no build complexity, good enough for a learning project

**JSON: nlohmann/json**
- single header, intuitive API, reads like Python dicts

**inference engine: llama.cpp**
- runs on CPU, quantized models, C API that's stable enough to build on
- wanted to build the inference server using C++

### response format
using OpenAI's completions API format (`choices[0].text`) so tark is a drop-in
compatible with any client that speaks OpenAI API

### error handling
400 on missing prompt, extracted into `validate_request()` util so both endpoints
reuse the same validation logic.

---

## token streaming via SSE

### SSE vs WebSockets vs long polling
**decision:** SSE (server-sent events)

- SSE: one-directional, server pushes to client, simple wire format (`data: {}\n\n`),
  works over plain HTTP, no handshake overhead
- rejected WebSockets: bidirectional (overkill for streaming tokens), more complex
  handshake, no benefit for this use case
- rejected long polling: client repeatedly polls, wastes connections, higher latency

### chunked transfer
cpp-httplib's `set_chunked_content_provider` handles chunked transfer encoding.
server skips Content-Length header, sends pieces, signals done. no manual implementation needed.

### endpoint design
added `/v1/completions/stream` as a separate endpoint rather than a `stream: true` flag
on `/v1/completions`. simpler to implement at this stage, flag-based switching comes
later when the two endpoints share more logic.

### done signal
`data: [DONE]\n\n` at end of stream. convention from OpenAI's API. return `false`
from chunked provider lambda to close the connection.

---

## KV cache reuse

### what is being reused
llama.cpp stores attention keys and values (KV cache) for every token processed.
on the next request, if the prompt shares a prefix with what's already cached,
we skip re-processing those tokens and start prefill from the first uncached token.

### edge cases
- empty cache: `pos_max` returns -1, so `start = 0`, full prefill
- cache extends beyond prompt (`start >= n_tokens`): new prompt is shorter than
  what's cached, or same prompt re-sent after generation. can't reuse, clear and restart.
- true reuse case: new prompt is a longer continuation of previous one (`start < n_tokens`).
  only prefill the new suffix.

### limitation
reuse only works within the same session on the same context. different requests
from different users don't share a context, each gets its own seq_id

---

## continuous batching

### problem with current design
one request at a time. HTTP handler calls `run_inference()` which blocks until done.
while one request decodes, every other request waits, GPU/CPU sits idle between requests

### HTTP layer design

**decision:** SSE from the start, single connection per request

three options considered:

1. **polling**: `POST` returns 202 + request_id, client polls `GET /result/{id}` until done.
   rejected: wastes connections, bad latency, annoying client implementation.

2. **two-step SSE**: `POST` returns 202 + request_id, client opens `GET /stream/{id}`.
   rejected: two connections per completion, extra client complexity, race condition
   between POST response and client opening the SSE connection.

3. **SSE from the start**: `POST /v1/completions/stream` keeps connection open immediately.
   scheduler picks it up, tokens stream back as generated, `[DONE]` when finished.
   chosen: one connection, maps to OpenAI's `stream:true` API, user sees tokens immediately.

### communication between scheduler and HTTP handler
HTTP handler can't run inference itself anymore, scheduler owns the decode loop.
need a way for the scheduler thread to push tokens to the HTTP thread.

**decision:** per-request token queue (std::queue + std::mutex + std::condition_variable)

HTTP handler:
1. parse prompt, create Request object with its own token_queue
2. push Request to scheduler's pending queue
3. loop: wait on condition_variable, read token from token_queue, write to SSE sink
4. when DONE signal arrives, send `[DONE]` and return

scheduler worker loop (runs on background thread):
1. each iteration: build one llama_batch with all active requests' current tokens
2. llama_decode once for all of them
3. sample one token per request, push to that request's token_queue
4. if EOS or max_tokens: push DONE signal, remove from active list
5. admit new requests from pending queue

### request struct
each request owns:
- `prompt_tokens`: tokenized input
- `output_tokens`: generated so far
- `seq_id`: unique ID for llama.cpp KV cache tracking
- `n_pos`: current position in sequence
- `done`: flag
- `token_queue` + `mutex` + `condition_variable`: communication channel to HTTP thread

### seq_id per request
previously all requests used `seq_id = 0`. with multiple concurrent requests,
each gets a unique seq_id. llama.cpp tracks KV cache separately per seq_id.
this is what makes multiple requests share one context without colliding.

### max batch size
context has fixed size (n_ctx = 2048). each active request consumes tokens.
scheduler needs to track total tokens in flight and cap active requests.
naive cap: `max_active = n_ctx / avg_expected_tokens`. tune later with benchmarks.

## code architecture refactoring

**decision:** split main.cpp into inference, server, scheduler and main files

**why:** main.cpp was becoming too complicated. as of now (before the refactor), it
had HTTP handlers, inference logic, and model setup. i'm planning on adding continuous
batching, which would make the code even more complex to understand and extend.

**file responsibilities:**
- inference.h/cpp: this will have all the interaction with llama.cpp and will be the
only file to do so. it'll have tokenize, build batch, prefill, decode, sample, etc
calls. will just have pure functions, no HTTP calls and no threading here.
- server.h/cpp: this will have all httplib code. it'll register endpoints, parse JSON,
validate the requests, build responses and so on. it calls the scheduler now, not
inference directly.
- scheduler.h/cpp: it has the Request struct, the Scheduler class, and the background
decode loop. it owns the pending queue, the active list, the per-request token queues,
and threading.
- main.h/cpp: it handles the startup of tark. contains init llama call, load model,
create scheduler, register routes, and listen calls. kind of like a table of contents.

**dependency direction:**
main -> scheduler -> inference -> llama.cpp
main -> server -> scheduler

also, this refactoring will improve the testability for inference logic.

## prefix caching

### goal
cache KV entries for a shared prefix once at startup. new requests that share the prefix copy those entries instead of reprocessing the prefix tokens.

### implementation
prefix stored on seq_id 0 via set_prefix() at startup. new requests call llama_memory_seq_cp() to copy prefix KV entries from seq_id 0 to their own seq_id, then only prefill the non-prefix portion.

### problem hit
seq_cp() asserted "only supported for full KV buffers" on partial range copies. by default, each seq_id gets its own separate memory buffer. copying between
two different buffers is a cross-stream copy, which requires moving actual memory bytes. llama.cpp only implements cross-stream copies for full buffers. partial ranges (p1 < buffer_size) fail the is_full check in llama-kv-cache.cpp.

### fix
cparams.kv_unified = true in main.cpp. puts all sequences into one shared buffer. seq_cp between sequences in the same buffer is now a metadata-only operation, just marks cells as belonging to the destination seq_id. no memory movement, no is_full check, partial ranges work.

### tradeoff
with separate streams, each seq_id has a dedicated fixed-size chunk. with unified, all sequences compete for one shared pool. for tark this is better since prefix slots are reused across requests rather than locked per-sequence.

### rejected
padding seq_id 0 with dummy tokens to satisfy the is_full check. hacky, wastes KV slots, doesn't scale.

### lesson
when hitting a library assert, read the source. the is_full check only exists on the cross-stream path (llama-kv-cache.cpp). same-stream path has no restriction. kv_unified forces same-stream.

## benchmark harness

added it to measure TTFT and total latency under concurrent load so i can compare
tark vs llama.cpp server side by side as architectural changes are made.

### stack
Python, httpx (async HTTP client), asyncio for concurrent requests.
httpx over aiohttp: cleaner API, actively maintained, same performance.

### output format
JSONL (one JSON object per run, appended to file). human-readable, git-diffable,
trivially extensible, same convention vLLM uses.

i take which server we are running the benchmark for, in the arguments, so a jsonl
output file is created for each server (only tark, and llama-server options are there, for now)
compare.py reads both, takes latest entry per concurrency level, prints side-by-side table.

### reason field
each run includes a reason field: a short note on why the benchmark was run.
passed via --reason flag, defaults to "unspecified". helps correlate performance
changes with architectural decisions when reviewing results later.

### error handling
two distinct failure modes, handled separately:
- ConnectError: server not running, connection refused
- 404: server running but wrong endpoint

previously both showed "All requests failed" with no distinction.

### rejected
SQLite: binary format, needs sqlite3 to inspect manually. JSONL is simpler and sufficient.
Avro: schema registry overhead, not human readable, overkill for a benchmark tool.