# what is an inference server
takes a model (weights on disk) and makes it serveable over a network.
handles HTTP, request parsing, running the model, streaming the response back.
tark is an inference server, llama.cpp is the engine inside it.

# llama.cpp
c++ library by Georgi Gerganov. runs LLMs locally on CPU via quantization.
loads a GGUF file (compressed weights + tokenizer vocab), exposes C API to tokenize/decode/sample.
so there's no need to write the transformer math, just the server around it.

# quantization
storing weights in 4-bit instead of 16-bit floats. ~4x smaller, small accuracy loss.
why it matters: LLaMA 7B in fp16 needs ~14GB VRAM. quantized fits in CPU RAM.
llama.cpp made this practical. enabled local LLM inference on consumer hardware.

# inference wrapper (milestone 1, part 1)
the c++ code that talks directly to llama.cpp and produces text.
no HTTP yet, just load model, run prompt, print tokens to stdout.

flow: backend_init -> model_load -> get_vocab -> tokenize -> batch prefill -> sampler_chain (greedy) -> decode loop -> token_to_piece.

vocab API takes llama_vocab* not llama_model* in recent llama.cpp versions.

# HTTP server (milestone 1, part 2)
wraps the inference wrapper in an HTTP server so anything (curl, client) can call it.
endpoint: POST /v1/completions. send a prompt, get generated text back as JSON.

cpp-httplib: single header HTTP server. svr.Post() registers a handler, req.body has raw JSON, res.set_content() sends response.
nlohmann/json: json::parse() for requests, json object construction for responses.
on Windows: must link ws2_32 (Winsock), without it all socket symbols are undefined at link time.

# prefill and decode
prefill: process the input prompt. all tokens fed to the model at once, in parallel. fast, builds the kv cache.
decode: generating the response, one token at a time, sequentially. each token depends on all previous ones so it cannot be parallelized. slow.
ttft (time to first token): time from request received to first generated token.
high because it includes the entire prefill step. longer prompt = higher TTFT.

# KV cache
created when context is initialized. stores attention states for every token seen so far.
lets the decode step skip recomputing the full sequence each time. just look up previous states.
grows with sequence length. memory bottleneck in production serving.

# SSE (server-sent events)
one HTTP request, connection stays open, server pushes tokens as they're sampled. the data is sent in chunks.
first the prompt tokens get batched and processed with llama_decode, then the logits are sampled for the last token from this. then for each subsequent token, a 1-sized batch is created and processed and the data is sent to the datasink using sink.write. at the end we send the data chunk wiht "[DONE]".
format: "data: {json}\n\n" per token, "data: [DONE]\n\n" at end.
gives the ChatGPT typewriter effect.

# greedy sampling
always pick the highest probability next token. deterministic, no randomness.
simplest possible sampler. production uses temperature + top-p instead.

# KV cache reuse
llama_memory_seq_pos_max returns the highest cached position for seq 0. we start the prefill from cached_pos+1 instead of 0. if the cache is empty, it returns -1.
if start >= n_tokens, cache extends beyond or exactly covers the new prompt. nothing left to prefill, can't reuse, so clear the cache and restart from 0. reuse only happens when the new prompt is longer than what's cached(start < n_tokens).

## prefix caching

goal: cache KV entries for a shared prefix once, reuse across requests. new requests copy prefix KV entries via llama_memory_seq_cp() and skip prefilling those tokens.

problem: seq_cp() asserted "only supported for full KV buffers" on partial range copies, i.e., when i filled a seq with just 12 tokens out of 256. by default, each seq_id gets its own separate memory buffer. copying between two different buffers requires moving actual memory bytes, and llama.cpp only implements this for full buffer copies. partial ranges (p1 < buffer_size) fail the is_full check in llama-kv-cache.cpp.

fix: cparams.kv_unified = true puts all sequences into one shared buffer. seq_cp between sequences in the same buffer is now just a metadata update, no memory movement, no is_full check, partial ranges work.

important lesson: read the source when hitting a library assert. the is_full check only exists on the cross-buffer path(between 2 different streams). forcing a shared buffer makes it disappear.

## benchmark harness

measure TTFT and total latency under concurrent load using async HTTP. asyncio.gather fires N requests simultaneously, each tracks time to first SSE token.

in tark: baseline on CPU + TinyLlama: 1.38s TTFT, 16.7s total at concurrency 4.

i accidentally wrote the default endpoint as /v1/completions in benchmark instead of the actual one, /v1/completions/stream and kept getting 'All requests failed' error message until i realised my mistake. i've added error handling for wrong url endpoint and server not running in the benchmark harness now.

#### note
silent failures are not good. in the above exception handling, i had just added the benchmark file so didn't take much time otherwise would've taken me a lot of time to figure out what the file is doing, then why this is wrong.
we should use exception handling as much as we can.

compare.py: last-write-wins per concurrency level, n/a for missing entries instead of crashing.

### benchmark run at concurrency 4 (3 runs x 4 concurrent, stable)

tark: 991ms TTFT, 16.5s total
llamacpp: 717ms TTFT, 14.4s total
llamacpp is ~28% faster on TTFT, ~13% faster on total latency.

root cause: tark calls llama_batch_init(active_list.size()) so batch shape changes as requests join/leave. llama.cpp rebuilds compute graph on every shape change. llamacpp server pads to fixed n_parallel=4, graph shape stays constant, reused every decode step ("graphs reused = 124" in logs).

to fix it, we will have to init batch to n_parallel size always, pad inactive slots