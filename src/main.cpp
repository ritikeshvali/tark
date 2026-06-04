#include "httplib.h"
#include "llama.h"
#include "scheduler.h"
#include "server.h"
#include <thread>

int main(int argc, char** argv) {
    if (argc<2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_seq_max = 8;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    Scheduler scheduler(model, ctx, vocab);

    std::thread scheduler_thread([&]() { scheduler.run(); });
    scheduler_thread.detach();

    httplib::Server server;
    register_routes(server, scheduler);

    fprintf(stdout, "tark listening on http://0.0.0.0:8080\n");
    server.listen("0.0.0.0", 8080);

    return 0;
}