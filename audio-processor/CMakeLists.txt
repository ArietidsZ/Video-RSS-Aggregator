cmake_minimum_required(VERSION 3.16)
project(AudioProcessor VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)

# Find packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# FFmpeg
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libavfilter
    libswresample
    libswscale
)

# Find NVIDIA Video Codec SDK (optional)
find_path(NVCODEC_INCLUDE_DIR nvEncodeAPI.h
    PATHS
    /usr/local/cuda/include
    /opt/nvidia/codec/include
    $ENV{NVCODEC_PATH}/include
)

find_library(NVCODEC_LIB
    NAMES nvidia-encode nvcuvid
    PATHS
    /usr/local/cuda/lib64
    /opt/nvidia/codec/lib
    $ENV{NVCODEC_PATH}/lib
)

if(NVCODEC_INCLUDE_DIR AND NVCODEC_LIB)
    message(STATUS "NVIDIA Video Codec SDK found")
    add_definitions(-DHAS_NVCODEC)
else()
    message(WARNING "NVIDIA Video Codec SDK not found - GPU acceleration disabled")
endif()

# WebRTC VAD
find_package(PkgConfig)
pkg_check_modules(WEBRTC webrtc-audio-processing)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FFMPEG_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVCODEC_INCLUDE_DIR}
    ${WEBRTC_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/audio_extractor.cpp
    src/gpu_accelerator.cpp
    src/audio_streamer.cpp
    src/audio_chunker.cpp
    src/vad_processor.cpp
    src/audio_normalizer.cpp
    src/ffmpeg_decoder.cpp
    src/websocket_server.cpp
    src/webrtc_handler.cpp
    src/audio_buffer.cpp
    src/performance_monitor.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/audio_kernels.cu
    src/cuda/resampling_kernels.cu
    src/cuda/normalization_kernels.cu
    src/cuda/vad_kernels.cu
)

# Create library
add_library(audio_processor SHARED ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(audio_processor
    PkgConfig::FFMPEG
    Threads::Threads
    CUDA::cudart
    CUDA::cufft
    CUDA::nppc
    ${NVCODEC_LIB}
)

# Set CUDA architecture
set_target_properties(audio_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Compiler flags
target_compile_options(audio_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Install targets
install(TARGETS audio_processor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/audio_processor
)

# Build examples
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Generate pkg-config file
configure_file(audio_processor.pc.in audio_processor.pc @ONLY)