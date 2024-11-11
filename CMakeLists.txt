
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(attension LANGUAGES CXX CUDA)

# Specify the C++ standard
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")
set(CMAKE_CUDA_ARCHITECTURES 75 86) 

# Specify the CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
# set nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --resource-usage -gencode arch=compute_86,code=sm_86")
#GDB
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g") 

# Find CUDA package
find_package(CUDA REQUIRED)

# find eigen
find_package(Eigen3 REQUIRED)

# add library
# file(GLOB ATTENTION_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/attention.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/self_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multi_head_self_attention.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flash_attention.cu)

target_include_directories(${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/cuda-samples/Common/)

# Link the CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# Collect all cuda test files
file(GLOB TEST_ATTENTION_SOURCES "test/*.cpp")
# Loop through each test source and create a separate executable
foreach(TEST_ATTENTION_SOURCE ${TEST_ATTENTION_SOURCES})
    get_filename_component(TEST_ATTENTION_NAME ${TEST_ATTENTION_SOURCE} NAME_WE)
    add_executable(${TEST_ATTENTION_NAME} ${TEST_ATTENTION_SOURCE})
    # target_include_directories(${TEST_ATTENTION_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TEST_ATTENTION_NAME} PRIVATE ${PROJECT_NAME})

    # Register each test with CTest
    enable_testing()
    add_test(NAME ${TEST_ATTENTION_NAME} COMMAND ${TEST_ATTENTION_NAME})
endforeach()
