add_executable(cuda_05 main.cu)

set_target_properties(cuda_05 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
