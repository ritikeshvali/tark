#include "server.h"
#include "inference.h"
#include "scheduler.h"

using json = nlohmann::json;

bool validate_request(const nlohmann::json &body, httplib::Response &res) {
    if (body.is_discarded() || !body.contains("prompt")) {
        res.status = 400;
        res.set_content(
            "{\"error\":\"prompt required\"}",
            "application/json"
        );
        return false;
    }
    return true;
}

void register_routes(httplib::Server& server, Scheduler& scheduler) {
    server.Get(
        "/health",
        [](const httplib::Request&,
            httplib::Response& res
        ) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    server.Post(
        "/v1/completions/stream",
        [&](const httplib::Request& req,
            httplib::Response& res
        ) {
            auto body = json::parse(req.body, nullptr, false);
            if (!validate_request(body, res)) return;

            std::string prompt = body["prompt"];

            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_chunked_content_provider("text/event-stream",
            [&, prompt](size_t, httplib::DataSink& sink) {
                auto req = scheduler.submit(prompt);

                while (true) {
                    std::unique_lock<std::mutex> lock(req->mtx);
                    req->cv.wait(lock, [&]{ return !req->token_queue.empty() || req->done; });
                    
                    while (!req->token_queue.empty()) {
                        llama_token tok = req->token_queue.front();
                        req->token_queue.pop();
                        lock.unlock();

                        // convert token to string and stream it
                        std::string piece = token_to_piece(scheduler.get_vocab(), tok);
                        std::string data = "data: " + nlohmann::json{{"token", piece}}.dump() + "\n\n";
                        sink.write(data.c_str(), data.size());

                        lock.lock();
                    }

                    if (req->done) {
                        sink.write("data: [DONE]\n\n", 14);
                        return false;
                    }
                }
            });
    });
}