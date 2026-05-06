cmake_minimum_required(VERSION 3.14)
project(tark)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(vendor/llama.cpp)

add_executable(tark src/main.cpp)
target_link_libraries(tark PRIVATE llama)
target_link_libraries(tark PRIVATE llama ws2_32)
target_include_directories(tark PRIVATE 
    vendor/llama.cpp/include
    vendor
)