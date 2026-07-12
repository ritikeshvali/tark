#include "scheduler.h"
#include "inference.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

std::shared_ptr<Request> Scheduler::submit(const std::string& prompt) {
    std::vector<llama_token> tokens = tokenize(this->vocab, prompt);

    auto req = std::make_shared<Request>();
    req->prompt_tokens = tokens;
    req->seq_id = next_seq_id_counter+1;
    next_seq_id_counter++;
    req->n_pos = 0;
    req->done = false;
    req->max_tokens = 128;

    {
        std::lock_guard<std::mutex> lock(pending_mtx);
        pending_queue.push(req);
    }

    return req;
}

void Scheduler::run() {
    auto sparams = llama_sampler_chain_default_params();

    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_sampler* draft_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(draft_sampler, llama_sampler_init_penalties(64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(draft_sampler, llama_sampler_init_greedy());

    while(true) {
        active_list.erase(
            std::remove_if(
                active_list.begin(),
                active_list.end(),
                [&](const std::shared_ptr<Request>& req) {
                    if (req->done) {
                        llama_memory_seq_rm(llama_get_memory(ctx), req->seq_id, -1, -1);
                        if (draft_ctx) {
                            llama_memory_seq_rm(llama_get_memory(draft_ctx), req->seq_id, -1, -1);
                        }
                        return true;
                    }
                    return false;
                }),
            active_list.end()
        );

        {
            std::lock_guard<std::mutex> lock(pending_mtx);
            while(!pending_queue.empty()) {
                auto req = pending_queue.front();
                pending_queue.pop();
                if (prefix_length>0 && req->prompt_tokens.size()>=prefix_length &&
                    std::equal(prefix_tokens.begin(), prefix_tokens.end(), req->prompt_tokens.begin())) {
                        llama_memory_seq_cp(llama_get_memory(ctx), 0, req->seq_id, 0, prefix_length);
                        llama_batch batch = llama_batch_init(req->prompt_tokens.size()-prefix_length, 0, 1);
                        run_prefill(ctx, batch, req->prompt_tokens, prefix_length, req->prompt_tokens.size(), req->seq_id);
                        req->n_pos = req->prompt_tokens.size();
                        llama_batch_free(batch);
                } else {
                    llama_batch batch = llama_batch_init(req->prompt_tokens.size(), 0, 1);
                    run_prefill(
                        this->ctx,
                        batch,
                        req->prompt_tokens,
                        0,
                        req->prompt_tokens.size(),
                        req->seq_id
                    );
                    req->n_pos = req->prompt_tokens.size();
                    llama_batch_free(batch);

                    if (draft_ctx) {
                        llama_batch draft_batch = llama_batch_init(req->prompt_tokens.size(), 0, 1);
                        run_prefill(draft_ctx, draft_batch, req->prompt_tokens, 0, req->prompt_tokens.size(), req->seq_id);
                        llama_batch_free(draft_batch);
                    }
                }
                this->active_list.push_back(req);
            }
        }

        if (active_list.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (draft_ctx) {
            const int N_DRAFT = 5;

            for (auto& req : active_list) {
                std::vector<llama_token> draft_tokens;
                llama_token last_tok = req->output_tokens.empty()
                                     ? req->prompt_tokens.back()
                                     : req->output_tokens.back();

                llama_batch verify_batch = llama_batch_init(N_DRAFT + 1, 0, 1);

                llama_batch seed = llama_batch_init(1, 0, 1);
                seed.token[0]     = last_tok;
                seed.pos[0]       = req->n_pos;
                seed.n_seq_id[0]  = 1;
                seed.seq_id[0][0] = req->seq_id;
                seed.logits[0]    = true;
                seed.n_tokens     = 1;

                if (llama_decode(draft_ctx, seed) != 0) {
                    fprintf(stderr, "draft seed decode failed\n");
                    llama_batch_free(seed);
                    llama_batch_free(verify_batch);
                    continue;
                }
                llama_batch_free(seed);

                llama_token draft_tok = llama_sampler_sample(draft_sampler, draft_ctx, 0);
                draft_tokens.push_back(draft_tok);

                for (int d = 1; d < N_DRAFT; d++) {
                    llama_batch db = llama_batch_init(1, 0, 1);
                    db.token[0]     = draft_tok;
                    db.pos[0]       = req->n_pos + d;
                    db.n_seq_id[0]  = 1;
                    db.seq_id[0][0] = req->seq_id;
                    db.logits[0]    = true;
                    db.n_tokens     = 1;
                    if (llama_decode(draft_ctx, db) != 0) {
                        llama_batch_free(db);
                        break;
                    }
                    llama_batch_free(db);
                    draft_tok = llama_sampler_sample(draft_sampler, draft_ctx, 0);
                    draft_tokens.push_back(draft_tok);
                }

                verify_batch.token[0]     = last_tok;
                verify_batch.pos[0]       = req->n_pos;
                verify_batch.n_seq_id[0]  = 1;
                verify_batch.seq_id[0][0] = req->seq_id;
                verify_batch.logits[0]    = false;
                verify_batch.n_tokens++;

                for (int d = 0; d < (int)draft_tokens.size(); d++) {
                    verify_batch.token[d+1]     = draft_tokens[d];
                    verify_batch.pos[d+1]       = req->n_pos + d + 1;
                    verify_batch.n_seq_id[d+1]  = 1;
                    verify_batch.seq_id[d+1][0] = req->seq_id;
                    verify_batch.logits[d+1]    = true;
                    verify_batch.n_tokens++;
                }

                if (llama_decode(this->ctx, verify_batch) != 0) {
                    fprintf(stderr, "llama_decode (verify) failed\n");
                    llama_batch_free(verify_batch);
                    continue;
                }

                int n_accepted = 0;
                for (int d = 0; d < (int)draft_tokens.size(); d++) {
                    llama_token target_tok = llama_sampler_sample(sampler, this->ctx, d + 1);
                    if (target_tok == draft_tokens[d]) {
                        n_accepted++;
                        std::lock_guard<std::mutex> lock(req->mtx);
                        req->token_queue.push(draft_tokens[d]);
                        req->cv.notify_one();
                        req->n_pos++;
                        req->output_tokens.push_back(draft_tokens[d]);
                        if (llama_vocab_is_eog(vocab, draft_tokens[d]) ||
                            (int)req->output_tokens.size() >= req->max_tokens) {
                            req->done = true;
                            break;
                        }
                    } else {
                        std::lock_guard<std::mutex> lock(req->mtx);
                        req->token_queue.push(target_tok);
                        req->cv.notify_one();
                        req->n_pos++;
                        req->output_tokens.push_back(target_tok);
                        if (llama_vocab_is_eog(vocab, target_tok) ||
                            (int)req->output_tokens.size() >= req->max_tokens) {
                            req->done = true;
                        }
                        break;
                    }
                }

                fprintf(stderr, "accepted %d/%zu draft tokens\n", n_accepted, draft_tokens.size());

                llama_memory_seq_rm(llama_get_memory(ctx), req->seq_id, req->n_pos, -1);
                llama_memory_seq_rm(llama_get_memory(draft_ctx), req->seq_id, req->n_pos, -1);

                llama_batch_free(verify_batch);
            }
        } else {
            llama_batch batch = llama_batch_init(active_list.size(), 0, 1);
            for (int i = 0; i < (int)active_list.size(); i++) {
                auto& req = active_list[i];
                batch.token[i]     = req->output_tokens.empty()
                                   ? req->prompt_tokens.back()
                                   : req->output_tokens.back();
                batch.pos[i]       = req->n_pos;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = req->seq_id;
                batch.logits[i]    = true;
                batch.n_tokens++;
            }

            if (llama_decode(this->ctx, batch) != 0) {
                fprintf(stderr, "llama_decode failed, skipping iteration\n");
                llama_batch_free(batch);
                continue;
            }

            for (int i = 0; i < (int)active_list.size(); i++) {
                auto& req = active_list[i];
                llama_token tok = llama_sampler_sample(sampler, this->ctx, i);
                {
                    std::lock_guard<std::mutex> lock(req->mtx);
                    req->token_queue.push(tok);
                }
                req->cv.notify_one();
                req->n_pos++;
                req->output_tokens.push_back(tok);
                if (llama_vocab_is_eog(vocab, tok) ||
                    (int)req->output_tokens.size() >= req->max_tokens) {
                    req->done = true;
                }
            }
            llama_batch_free(batch);
        }
    }
}

void Scheduler::set_prefix(const std::string& prefix_text) {
    prefix_tokens = tokenize(vocab, prefix_text);
    prefix_length = prefix_tokens.size();

    llama_batch batch = llama_batch_init(prefix_length, 0, 1);
    run_prefill(ctx, batch, prefix_tokens, 0, prefix_length, 0);
    llama_batch_free(batch);
}