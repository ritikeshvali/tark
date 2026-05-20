#include "llama.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// all your inference in one function
std::string run_inference(llama_model* model, llama_context* ctx, const std::string& prompt) {
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(), true, false
    );
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1);
        batch.n_tokens++;
    }

    llama_decode(ctx, batch);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string result;
    char piece[128];
    int n_cur = n_tokens;
    int n_max = 128;

    while (n_cur < n_max) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;

        int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        if (n > 0) { piece[n] = '\0'; result += piece; }

        llama_batch_free(batch);
        batch = llama_batch_init(1, 0, 1);
        batch.token[0]     = tok;
        batch.pos[0]       = n_cur;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;

        llama_decode(ctx, batch);
        n_cur++;
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return result;
}

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

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("prompt")) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt required\"}", "application/json");
            return;
        }

        std::string prompt = body["prompt"];
        std::string output = run_inference(model, ctx, prompt);

        json response = {
            {"id", "cmpl-1"},
            {"object", "text_completion"},
            {"model", "tark"},
            {"choices", json::array({{
                {"text", output},
                {"finish_reason", "stop"}
            }})}
        };

        res.set_content(response.dump(), "application/json");
    });

    fprintf(stdout, "tark listening on http://0.0.0.0:8080\n");
    svr.listen("0.0.0.0", 8080);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}