#pragma once
#include "llama.h"
#include <vector>
#include <string>

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& prompt);

std::string token_to_piece(const llama_vocab* vocab, llama_token token);

void run_prefill(
    llama_context* ctx, llama_batch& batch, const std::vector<llama_token>& tokens,
    int start, int n_tokens, int seq_id
);