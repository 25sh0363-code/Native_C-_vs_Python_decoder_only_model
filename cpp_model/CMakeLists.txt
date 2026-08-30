cmake_minimum_required(VERSION 3.18)
project(nanogpt_libtorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at your libtorch install, e.g.:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)

add_library(gptcore
    src/gpt_model.cpp
    src/bin_dataset.cpp
)
target_link_libraries(gptcore "${TORCH_LIBRARIES}")

add_executable(train src/train.cpp)
target_link_libraries(train gptcore "${TORCH_LIBRARIES}")

add_executable(generate src/generate.cpp)
target_link_libraries(generate gptcore "${TORCH_LIBRARIES}")

set_property(TARGET train PROPERTY CXX_STANDARD 17)
set_property(TARGET generate PROPERTY CXX_STANDARD 17)