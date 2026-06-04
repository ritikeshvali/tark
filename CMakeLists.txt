cmake_minimum_required(VERSION 3.14)
project(tark)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(vendor/llama.cpp)

add_definitions(-D_WIN32_WINNT=0x0A00)
add_executable(tark
    src/main.cpp
    src/inference.cpp
    src/server.cpp
    src/scheduler.cpp
)

target_link_libraries(tark PRIVATE llama ws2_32)
target_include_directories(tark PRIVATE
    vendor/llama.cpp/include
    vendor
    src
)