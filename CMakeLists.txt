cmake_minimum_required(VERSION 3.20)
project(gpu-geometry-benchmarks VERSION 0.1.0 LANGUAGES CXX CUDA)

add_subdirectory(src)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT main)
