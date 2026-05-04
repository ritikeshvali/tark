# tark

a C++ LLM inference server written from scratch.

minimal, deterministic, and built close to the metal. focused on correctness, performance, and composable primitives.

## why "tark"?
tark mean logic or reasoning, deriving outputs through structured steps. this system does the same, executing inference as a sequence of explicit, controllable operations.

## status
in active development

## current capabilities

## in progress

## planned capabilities
- minimum viable HTTP server with single-request inference
- token streaming (SSE)
- KV cache reuse across turns
- continuous batching
- prefix caching
- benchmark harness

## stack
- C++ 17
- llama.cpp (engine)
- cpp-httplib (HTTP)
- nlohmann/json (request/response parsing)

## notes
design notes and implementation details live in [NOTES.md](./NOTES.md)