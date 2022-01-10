add_executable(cuda_03 main.cu)

set_target_properties(cuda_03 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
