cmake_minimum_required(VERSION 3.20)
project(learn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

file (GLOB_RECURSE _files "${CMAKE_CURRENT_SOURCE_DIR}/*.*")

message($_file)

include_directories(include)
add_subdirectory(cuda_03)
add_subdirectory(cuda_04)
add_subdirectory(cuda_05)
#add_subdirectory(cuda_06)


