add_executable(cuda_04 main.cu)

set_target_properties(cuda_04 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
