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
    req->seq_id = next_seq_id_counter++;
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
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    while(true) {
        active_list.erase(
            std::remove_if(
                active_list.begin(),
                active_list.end(),
                [&](const std::shared_ptr<Request>& req) {
                    if (req->done) {
                        llama_memory_seq_rm(llama_get_memory(ctx), req->seq_id, -1, -1);
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
                this->active_list.push_back(req);
            }
        }

        if (active_list.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        llama_batch batch = llama_batch_init(active_list.size(), 0, 1);
        for (int i=0; i<active_list.size(); i++) {
            auto& req = active_list[i];
            batch.token[i] = req->output_tokens.empty()
                            ? req->prompt_tokens.back()
                            : req->output_tokens.back();
            batch.pos[i] = req->n_pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = req->seq_id;
            batch.logits[i] = true;
            batch.n_tokens++;
        }

        llama_decode(this->ctx, batch);

        for (int i=0; i<active_list.size(); i++) {
            auto& req = active_list[i];
            llama_token tok =  llama_sampler_sample(sampler, this->ctx, i);
            
            {
                std::lock_guard<std::mutex> lock(req->mtx);
                req->token_queue.push(tok);
            }
            req->cv.notify_one();   // only one HTTP thread waits per request
            req->n_pos++;
            req->output_tokens.push_back(tok);
            if (llama_vocab_is_eog(vocab, tok) || (int)req->output_tokens.size() >= req->max_tokens) {
                req->done = true;
            }
        }
        llama_batch_free(batch);
    }
}
