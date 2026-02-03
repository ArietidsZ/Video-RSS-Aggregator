cmake_minimum_required(VERSION 3.23)
project(vra_backends LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VRA_BACKEND_KIND "cpu" CACHE STRING "Backend kind: cpu|mps|coreml|cuda|rocm|oneapi")
string(TOLOWER "${VRA_BACKEND_KIND}" VRA_BACKEND_KIND_LOWER)

set(WHISPER_CPP_INCLUDE_DIR "" CACHE PATH "Path to whisper.cpp include directory")
set(WHISPER_CPP_LIB_DIR "" CACHE PATH "Path to whisper.cpp library directory")
set(LLAMA_CPP_INCLUDE_DIR "" CACHE PATH "Path to llama.cpp include directory")
set(LLAMA_CPP_LIB_DIR "" CACHE PATH "Path to llama.cpp library directory")

if (WHISPER_CPP_INCLUDE_DIR STREQUAL "" OR WHISPER_CPP_LIB_DIR STREQUAL "")
    message(FATAL_ERROR "WHISPER_CPP_INCLUDE_DIR and WHISPER_CPP_LIB_DIR must be set")
endif ()

if (LLAMA_CPP_INCLUDE_DIR STREQUAL "" OR LLAMA_CPP_LIB_DIR STREQUAL "")
    message(FATAL_ERROR "LLAMA_CPP_INCLUDE_DIR and LLAMA_CPP_LIB_DIR must be set")
endif ()

find_library(WHISPER_LIB whisper PATHS ${WHISPER_CPP_LIB_DIR} NO_DEFAULT_PATH)
find_library(LLAMA_LIB llama PATHS ${LLAMA_CPP_LIB_DIR} NO_DEFAULT_PATH)

if (NOT WHISPER_LIB)
    message(FATAL_ERROR "Could not find whisper library in ${WHISPER_CPP_LIB_DIR}")
endif ()

if (NOT LLAMA_LIB)
    message(FATAL_ERROR "Could not find llama library in ${LLAMA_CPP_LIB_DIR}")
endif ()

add_library(vra_backend SHARED src/vra_backend.cpp)
target_include_directories(vra_backend PRIVATE ${WHISPER_CPP_INCLUDE_DIR} ${LLAMA_CPP_INCLUDE_DIR})
target_link_libraries(vra_backend PRIVATE ${WHISPER_LIB} ${LLAMA_LIB})
target_compile_definitions(vra_backend PRIVATE VRA_BACKEND_KIND_STR="${VRA_BACKEND_KIND_LOWER}")

set_target_properties(vra_backend PROPERTIES OUTPUT_NAME "vra_${VRA_BACKEND_KIND_LOWER}_backend")

if (APPLE)
    target_link_libraries(vra_backend PRIVATE "-framework Foundation")
    if (VRA_BACKEND_KIND_LOWER STREQUAL "mps")
        target_link_libraries(vra_backend PRIVATE "-framework Metal" "-framework MetalPerformanceShaders")
    endif ()
    if (VRA_BACKEND_KIND_LOWER STREQUAL "coreml")
        target_link_libraries(vra_backend PRIVATE "-framework CoreML")
    endif ()
endif ()
