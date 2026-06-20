# local.sh: made this helper file with functions for running tark locally
# requires .env file with MODEL set
TARK_DIR="/c/Users/ritik/OneDrive/Documents/tark"
source "$TARK_DIR/.env"

start_server() {
    cd "$TARK_DIR/build"
    ./tark.exe "$MODEL"
}

start_llamacpp() {
    "$TARK_DIR/vendor/llama.cpp/build/bin/llama-server.exe" \
        -m "$MODEL" --port 8081
}

run_bench() {
    cd "$TARK_DIR"
    python bench/benchmark.py --server tark --concurrency 4 --reason "${1:-unspecified}" "${@:2}"
}

bench_llamacpp() {
    cd "$TARK_DIR"
    python bench/benchmark.py --server llamacpp \
        --url http://localhost:8081/completion \
        --concurrency 4 \
        --reason "${1:-baseline}" "${@:2}"
}

# run an instance of msys2 mingw64
# run - source local.sh
# then run the func needed - start_server/start_llamacpp