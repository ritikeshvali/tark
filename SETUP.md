# setup

## prerequisites
- CMake 3.14+
- GCC 13+ (or Clang with C++17 support)
- On Windows: MSYS2 with mingw-w64-x86_64 toolchain

## clone and init

```bash
git clone https://github.com/ritikeshvali/tark.git
cd tark
git submodule update --init --recursive
```

the last command pulls llama.cpp — it's a submodule, not vendored directly. takes a minute.

## build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

on Windows (MSYS2 MINGW64 terminal):
```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
```

## run

```bash
./tark /path/to/model.gguf
```

you need a GGUF model file. recommended for testing:
`TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` from HuggingFace — small enough for CPU.

model files are gitignored. download separately.

## test

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 128}'
```