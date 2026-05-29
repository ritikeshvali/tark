# tark

a C++ LLM inference server written from scratch.

minimal, deterministic, and built close to the metal. focused on correctness, performance, and composable primitives.

## why "tark"?
tark means logic or reasoning, deriving outputs through structured steps. this system does the same, executing inference as a sequence of explicit, controllable operations.

## status
in active development

## current capabilities
- HTTP server with POST /v1/completions endpoint
- single-request inference via llama.cpp
- greedy decoding with GGUF model support
- token streaming (SSE)
- KV cache reuse across turns

## in progress
- continuous batching

## planned capabilities
- prefix caching
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

## notes
design notes and implementation details live in [NOTES.md](./NOTES.md)