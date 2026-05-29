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