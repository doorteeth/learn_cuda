add_executable(cuda_08 main.cu)

set_target_properties(cuda_08 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

