find_package(OpenCL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
set_target_properties(gtest PROPERTIES FOLDER third_party)
set_target_properties(gtest_main PROPERTIES FOLDER third_party)

add_executable(main
    main.cpp
    buckets.h
    buckets.cpp
    clDeviceManager.h
    clDeviceManager.cpp
    clUtils.h
    clUtils.cpp
    cpu.h
    cpu.cpp
    gpu.h
    gpu.cpp
    structs.h
    structs.cpp
    timer.h
    timer.cpp
    kernel.cu
)

set_target_properties(main PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  CUDA_ARCHITECTURES native
)

target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main PRIVATE
  OpenCL::OpenCL
  CUDA::cudart_static
  GTest::gtest_main
)
add_compile_definitions(CL_TARGET_OPENCL_VERSION=120)

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -Wno-deprecated-gpu-targets;
  -Xptxas=-v;
  #-G;
  #-keep;
>)

if (MSVC)
  target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /permissive /arch:AVX2 >)
else()
  target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Wpedantic -mavx2 -march=native >)
endif()

include(GoogleTest)
gtest_discover_tests(main)
