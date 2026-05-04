# inference wrapper (milestone 1)
llama.cpp flow: backend_init -> model_load -> get_vocab -> tokenize -> batch prefill -> sampler_chain (greedy) -> decode loop -> token_to_piece.
vocab API takes llama_vocab* not llama_model* in recent llama.cpp versions.