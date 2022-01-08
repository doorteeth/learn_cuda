cmake_minimum_required(VERSION 3.20)
project(learn_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(learn_cuda main.cu)

set_target_properties(learn_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
