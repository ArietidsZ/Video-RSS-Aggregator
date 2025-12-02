cmake_minimum_required(VERSION 3.18)
project(whisper_turbo VERSION 1.0.0 LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
if(USE_CUDA)
    find_package(CUDA REQUIRED)
endif()
find_package(Threads REQUIRED)
find_package(OpenMP)

# Find or fetch CTranslate2 (Faster Whisper backend)
include(FetchContent)
FetchContent_Declare(
    ctranslate2
    GIT_REPOSITORY https://github.com/OpenNMT/CTranslate2.git
    GIT_TAG v3.20.0
)
FetchContent_MakeAvailable(ctranslate2)

# Find or fetch oneDNN for CPU optimizations
FetchContent_Declare(
    onednn
    GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
    GIT_TAG v3.3
)
FetchContent_MakeAvailable(onednn)

if(USE_CUDA)
    # CUDA architecture detection
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_LIST ${INSTALLED_GPU_CCS})
    list(REMOVE_DUPLICATES INSTALLED_GPU_CCS_LIST)

    # Set CUDA flags for detected architectures
    set(CMAKE_CUDA_ARCHITECTURES ${INSTALLED_GPU_CCS_LIST})
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")

if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_CUDA)
    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -march=native")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v")
endif()

# Add whisper-turbo library
set(WHISPER_SOURCES
    src/whisper_turbo.cpp
    src/batch_processor.cpp
    src/chinese_asr.cpp
    src/quantization.cpp
    src/streaming_engine.cpp
)

if(USE_CUDA)
    list(APPEND WHISPER_SOURCES
        src/cuda/attention_kernels.cu
        src/cuda/batch_kernels.cu
        src/cuda/fused_kernels.cu
        src/cuda/graph_executor.cu
        src/cuda/quantize_kernels.cu
    )
endif()

add_library(whisper_turbo SHARED ${WHISPER_SOURCES})

target_include_directories(whisper_turbo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ctranslate2_SOURCE_DIR}/include
    ${onednn_SOURCE_DIR}/include
)

target_link_libraries(whisper_turbo
    ${CMAKE_THREAD_LIBS_INIT}
    ctranslate2
    dnnl
)

if(USE_CUDA)
    target_link_libraries(whisper_turbo
        ${CUDA_LIBRARIES}
        cublas
        cublasLt
        cudnn
        cufft
        curand
        cusparse
        nvrtc
    )
endif()

if(OpenMP_FOUND)
    target_link_libraries(whisper_turbo OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    # Enable CUDA graphs for kernel fusion
    target_compile_definitions(whisper_turbo PUBLIC
        USE_CUDA_GRAPH=1
        USE_TENSOR_CORES=1
        USE_FLASH_ATTENTION=1
        USE_FP8_QUANTIZATION=1
    )
endif()

# Build benchmarks


# Install rules
install(TARGETS whisper_turbo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Generate pkg-config file
configure_file(whisper-turbo.pc.in whisper-turbo.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/whisper-turbo.pc
    DESTINATION lib/pkgconfig
)