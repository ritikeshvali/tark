#pragma once
#include "nlohmann/json.hpp"
#include "httplib.h"
#include "scheduler.h"

bool validate_request(
    const nlohmann::json& body,
    httplib::Response& res
);

void register_routes(httplib::Server& server, Scheduler& scheduler);