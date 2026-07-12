#include "httplib.h"
#include "llama.h"
#include "scheduler.h"
#include "server.h"
#include <cstdio>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <target_model.gguf> [draft_model.gguf]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    // load target model
    auto mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "Failed to load target model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_seq_max = 8;
    cparams.kv_unified = true;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create target context\n");
        return 1;
    }

    // load draft model (optional)
    llama_model* draft_model = nullptr;
    llama_context* draft_ctx = nullptr;

    if (argc >= 3) {
        draft_model = llama_model_load_from_file(argv[2], mparams);
        if (!draft_model) {
            fprintf(stderr, "Failed to load draft model\n");
            return 1;
        }

        auto draft_cparams = llama_context_default_params();
        draft_cparams.n_ctx = 2048;
        draft_cparams.n_batch = 512;
        draft_cparams.n_seq_max = 8;
        draft_cparams.kv_unified = true;
        draft_ctx = llama_init_from_model(draft_model, draft_cparams);
        if (!draft_ctx) {
            fprintf(stderr, "Failed to create draft context\n");
            return 1;
        }

        fprintf(stdout, "speculative decoding enabled\n");
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    Scheduler scheduler(model, ctx, vocab, draft_model, draft_ctx);

    std::thread scheduler_thread([&]() { scheduler.run(); });
    scheduler_thread.detach();

    httplib::Server server;
    register_routes(server, scheduler);

    fprintf(stdout, "tark listening on http://0.0.0.0:8080\n");
    server.listen("0.0.0.0", 8080);

    llama_free(ctx);
    llama_model_free(model);
    if (draft_ctx) llama_free(draft_ctx);
    if (draft_model) llama_model_free(draft_model);
    llama_backend_free();

    return 0;
}