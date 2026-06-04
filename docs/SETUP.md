# setup and usage

## prerequisites
- CMake 3.14+
- GCC 13+ via MSYS2 (mingw-w64-x86_64 toolchain)
- On Windows: MSYS2 with MinGW64 terminal

## model
download a GGUF model to run. recommended for testing:

**TinyLlama 1.1B Q4_K_M:** small enough for CPU, good for testing
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

model files are gitignored. download separately and pass the path at startup.

## build

```bash
git clone https://github.com/ritikeshvali/tark.git
cd tark
git submodule update --init --recursive
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
```

## run

```bash
./tark /path/to/model.gguf
```

server starts on http://0.0.0.0:8080

## endpoints

### health check
```bash
curl http://localhost:8080/health
```

### streaming inference
```bash
curl -X POST http://localhost:8080/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is"}' \
  --no-buffer
```

tokens stream back as they are sampled, one per SSE event:
```
data: {"token":" to"}
data: {"token":" be"}
data: {"token":" yourself"}
...
data: [DONE]
```

### concurrent requests via continuous batching
fire two requests simultaneously to see continuous batching in action.
tokens from both requests interleave in the output as the scheduler decodes them together.

```bash
curl -X POST http://localhost:8080/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is"}' \
  --no-buffer &

curl -X POST http://localhost:8080/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}' \
  --no-buffer
```

expected output, tokens from both prompts interleaved:
```
data: {"token":" to"}
data: {"token":" Paris"}
data: {"token":" be"}
data: {"token":"."}
...
```

## context size
default context size is 2048 tokens across all active sequences.
supports up to 8 concurrent sequences (n_seq_max=8).
