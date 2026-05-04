#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 2048;
    cparams.n_batch = 512;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);  // add this

    std::string prompt = "The meaning of life is";
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),  // vocab here
        tokens.data(), tokens.size(), true, false
    );

    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(cparams.n_batch, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1);
        batch.n_tokens++;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill decode failed\n");
        return 1;
    }

    // sampler setup
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // generation loop
    int n_cur = n_tokens;
    int n_max = 128;
    char piece[128];

    while (n_cur < n_max) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, tok)) break;
        
        int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        if (n > 0) {
            piece[n] = '\0';
            printf("%s", piece);
            fflush(stdout);
        }

        llama_batch_free(batch);
        batch = llama_batch_init(1, 0, 1);
        batch.token[0]     = tok;
        batch.pos[0]       = n_cur;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Decode failed at token %d\n", n_cur);
            break;
        }
        n_cur++;
    }

    printf("\n");
    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}