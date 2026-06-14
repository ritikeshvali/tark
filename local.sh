# local.sh: made this helper file with functions for running tark locally
# requires .env file with MODEL set
source "$(dirname "$0")/.env"

start_server() {
    cd "$(dirname "$0")/build"
    ./tark.exe "$MODEL"
}

start_llamacpp() {
    "$(dirname "$0")/vendor/llama.cpp/build/bin/llama-server.exe" \
        -m "$MODEL" --port 8081
}

run_bench() {
    cd "$(dirname "$0")"
    python bench/benchmark.py --server tark --concurrency 4 --reason "${1:-unspecified}" "${@:2}"
}

bench_llamacpp() {
    cd "$(dirname "$0")"
    python bench/benchmark.py --server llamacpp \
        --url http://localhost:8081/completion \
        --concurrency 4 \
        --reason "${1:-baseline}" "${@:2}"
}

# run an instance of msys2 mingw64
# run - source local.sh
# then run the func needed - start_server/start_llamacpp