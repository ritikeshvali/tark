#include "inference.h"
#include "llama.h"

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& prompt) {
    std::vector<llama_token> tokens(prompt.size()+32);
    int n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(),
        true, false
    );
    tokens.resize(n_tokens);

    return tokens;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    char piece[128];
    int n = llama_token_to_piece(
        vocab, token, piece, sizeof(piece),
        0, false
    );
    if (n>0) {
        piece[n] = '\0';
        return std::string(piece);
    }
    return "";
}

void run_prefill(
    llama_context* ctx,
    llama_batch& batch,
    const std::vector<llama_token>& tokens,
    int start,
    int n_tokens,
    int seq_id) {
        int batch_idx = 0;
        for (int i=start; i<n_tokens; i++) {
            batch.token[batch_idx] = tokens[i];
            batch.pos[batch_idx] = i;
            batch.n_seq_id[batch_idx] = 1;  // how many seq this token belongs to
            batch.seq_id[batch_idx][0] = seq_id;    // which seq IDs this token belongs to, 2D array
            batch.logits[batch_idx] = (i==n_tokens-1);
            batch.n_tokens++;
            batch_idx++;
        }

        llama_decode(ctx, batch);
}