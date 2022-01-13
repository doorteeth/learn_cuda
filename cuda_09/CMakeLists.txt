
add_executable(cuda_09 main.cu)

set_target_properties(cuda_09 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)