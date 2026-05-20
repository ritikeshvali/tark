# inference wrapper (milestone 1)
llama.cpp flow: backend_init -> model_load -> get_vocab -> tokenize -> batch prefill -> sampler_chain (greedy) -> decode loop -> token_to_piece.
vocab API takes llama_vocab* not llama_model* in recent llama.cpp versions.

# HTTP server (milestone 1 complete)
cpp-httplib: single header HTTP server. svr.Post() registers a handler, req.body has raw JSON, res.set_content() sends response.
on Windows, must link ws2_32 (Winsock), without it all socket symbols are undefined at link time.
nlohmann/json: json::parse() for requests, json object construction for responses.

# prefill and decode
prefill: process the input prompt. all tokens fed to the model at once, in parallel. fast, builds the kv cache.
decode: generating the response, one token at a time, sequentially. each token depends on all previous ones so it cannot be parallelized. slow.
ttft (time to first token): time from request received to first generated token.
high because it includes the entire prefill step. longer prompt = higher TTFT.