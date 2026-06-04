# tark

a C++ LLM inference server written from scratch.

minimal, deterministic, and built close to the metal. focused on correctness, performance, and composable primitives.

## why "tark"?
tark means logic or reasoning, deriving outputs through structured steps. this system does the same, executing inference as a sequence of explicit, controllable operations.

## how it works
HTTP thread       ->  submit request to scheduler
scheduler thread  ->  one llama_decode per iteration across all active requests
                  ->  push sampled tokens to per-request queue
HTTP thread       ->  stream tokens to client via SSE

## demo

**server startup:**

![server](assets/server.gif)

**concurrent requests, tokens interleaving:**

![requests](assets/requests.gif)

## current capabilities
- HTTP server with POST /v1/completions endpoint
- single-request inference via llama.cpp
- greedy decoding with GGUF model support
- token streaming (SSE)
- KV cache reuse across turns
- continuous batching

## in progress
- prefix caching

## planned capabilities
- benchmark harness

## stack
- C++17
- llama.cpp (engine)
- cpp-httplib (HTTP)
- nlohmann/json (request/response parsing)
- chrono (time durations)

## running

**start the server**
```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
./tark /path/to/model.gguf
```

**test endpoints**
```bash
# health check
curl http://localhost:8080/health

# single response
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is"}'

# streaming
curl -X POST http://localhost:8080/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is"}' \
  --no-buffer
```

## docs
- [setup and usage](docs/SETUP.md)
- [design decisions](docs/DESIGN.md)

## notes
design notes and implementation details live in [NOTES.md](./NOTES.md)