#pragma once
#include "llama.h"
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

struct Request {
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> output_tokens;
    int seq_id;
    int n_pos;
    bool done;
    int max_tokens = 128;  // default

    // the communication channel: token queue + mutex + condition variable
    std::queue<llama_token> token_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

class Scheduler {
    std::queue<std::shared_ptr<Request>> pending_queue;
    std::vector<std::shared_ptr<Request>> active_list;
    std::vector<llama_token> prefix_tokens;
    std::mutex pending_mtx;
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;
    int next_seq_id_counter;
    int prefix_length = 0;

public:
    std::shared_ptr<Request> submit(const std::string& prompt);
    void run();
    const llama_vocab* get_vocab() const { return vocab; }

    Scheduler(llama_model* model, llama_context* ctx, const llama_vocab* vocab)
    : model(model), ctx(ctx), vocab(vocab), next_seq_id_counter(0) {}

    void set_prefix(const std::string& prefix_text);
};