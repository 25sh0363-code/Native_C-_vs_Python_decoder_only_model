cmake_minimum_required(VERSION 3.18)
project(gpt_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at your extracted LibTorch directory, e.g.:
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(gpt_core STATIC
    src/gpt_model.cpp
    src/bpe_tokenizer.cpp
)
target_include_directories(gpt_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpt_core PUBLIC "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_compile_options(gpt_core PRIVATE ${TORCH_CXX_FLAGS})

add_executable(train src/train.cpp)
target_link_libraries(train PRIVATE gpt_core "${TORCH_LIBRARIES}")

add_executable(generate src/generate.cpp)
target_link_libraries(generate PRIVATE gpt_core "${TORCH_LIBRARIES}")

add_executable(tokenize_data src/tokenize_data.cpp)
target_link_libraries(tokenize_data PRIVATE gpt_core "${TORCH_LIBRARIES}")

set_property(TARGET train generate tokenize_data PROPERTY CXX_STANDARD 17)

# LibTorch ships its own DLLs on Windows; copy them next to the executables.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(target train generate tokenize_data)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${target}>)
    endforeach()
endif()
