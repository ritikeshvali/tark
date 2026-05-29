#include "llama.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <chrono>

using json = nlohmann::json;

// util - validate prompt request
bool validate_request (const json& body, httplib::Response& res) {
    if (body.is_discarded() || !body.contains("prompt")) {
        res.status = 400;
        res.set_content("{\"error\":\"prompt required\"}", "application/json");
        return false;
    }
    return true;
}

// this has all the inference code
std::string run_inference(llama_model* model, llama_context* ctx, const std::string& prompt) {
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(), true, false
    );
    tokens.resize(n_tokens);

    // llama_memory_clear(llama_get_memory(ctx) , false);
    llama_pos cached_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    int start = (cached_pos<0)? 0: cached_pos+1;

    if (start >= n_tokens) {
        llama_memory_clear(llama_get_memory(ctx), false);
        start = 0;
    }
    
    llama_batch batch = llama_batch_init(512, 0, 1);
    int batch_idx = 0;
    for (int i = start; i < n_tokens; i++) {
        batch.token[batch_idx] = tokens[i];
        batch.pos[batch_idx] = i;
        batch.n_seq_id[batch_idx] = 1;
        batch.seq_id[batch_idx][0] = 0;
        batch.logits[batch_idx] = (i == n_tokens - 1);
        batch.n_tokens++;
        batch_idx++;
    }

    llama_decode(ctx, batch);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string result;
    char piece[128];
    int n_cur = n_tokens;
    int n_max = n_tokens + 128;

    while (n_cur < n_max) {
        llama_token tok =  llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;

        int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        if (n>0) {
            piece[n] = '\0';
            result += piece;
        }

        llama_batch_free(batch);
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = tok;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        llama_decode(ctx, batch);
        n_cur++;
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return result;
}

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
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context");
        return 1;
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body, nullptr, false);
        if (!validate_request(body, res))
            return;

        std::string prompt = body["prompt"];
        
        auto t_start = std::chrono::steady_clock::now();
        std::string output = run_inference(model, ctx, prompt);
        auto t_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stdout, "inference: %.1f ms\n", ms);
        fflush(stdout);

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

    svr.Post("/v1/completions/stream", [&](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body, nullptr, false);
        if (!validate_request(body, res))
            return;

        std::string prompt = body["prompt"];

        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider("text/event-stream",
        [&, prompt](size_t, httplib::DataSink& sink) {
            const llama_vocab* vocab = llama_model_get_vocab(model);

            std::vector<llama_token> tokens(prompt.size()+32);
            int n_tokens = llama_tokenize(
                vocab, prompt.c_str(), prompt.size(),
                tokens.data(), tokens.size(), true, false
            );
            tokens.resize(n_tokens);

            // llama_memory_clear(llama_get_memory(ctx) , false);
            llama_pos cached_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            int start = (cached_pos<0)? 0: cached_pos+1;

            if (start>=n_tokens) {
                llama_memory_clear(llama_get_memory(ctx), false);
                start = 0;
            }

            // we batch all the prompt tokens, uptil 512
            llama_batch batch = llama_batch_init(512, 0, 1);
            int batch_idx = 0;
            for (int i=start; i<n_tokens; i++) {
                batch.token[batch_idx] = tokens[i];
                batch.pos[batch_idx] = i;
                batch.n_seq_id[batch_idx] = 1;
                batch.seq_id[batch_idx][0] = 0;
                batch.logits[batch_idx] = (i == n_tokens-1);
                batch.n_tokens++;
                batch_idx++;
            }
            // we do the prefil step when we call llama_decode
            llama_decode(ctx, batch);

            // sampling selects the next token from logits after a decode call
            auto sparams = llama_sampler_chain_default_params();
            // there's many strategies like greedy, temperature, top-p, top-k
            llama_sampler* sampler = llama_sampler_chain_init(sparams);
            // we choose greedy strategy here
            // in prod we chain strategies, eg: temp first, then top-k, then pick
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

            char piece[128];
            int n_cur = n_tokens;
            int n_max = n_tokens + 128;

            while (n_cur < n_max) {
                // -1 because after prefill, only the last token had logits created for it
                llama_token tok = llama_sampler_sample(sampler, ctx, -1);
                if (llama_vocab_is_eog(vocab, tok))
                    break;

                    int n = llama_token_to_piece(
                        vocab, tok, piece, sizeof(piece), 0, false);
                    if (n>0) {
                        piece[n] = '\0';
                        json event = {{"token", std::string(piece)}};
                        std::string data = "data: " + event.dump() + "\n\n";
                        sink.write(data.c_str(), data.size());
                    }

                    llama_batch_free(batch);
                    batch = llama_batch_init(1, 0, 1);
                    batch.token[0] = tok;
                    batch.pos[0] = n_cur;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = 0;
                    batch.logits[0] = true;
                    batch.n_tokens = 1;
                    llama_decode(ctx, batch);
                    n_cur++;
            }

            std::string done = "data: [DONE]\n\n";
            sink.write(done.c_str(), done.size());

            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return false;
        });
    });

    fprintf(stdout, "tark listening on http://0.0.0.0:8080\n");
    svr.listen("0.0.0.0", 8080);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}