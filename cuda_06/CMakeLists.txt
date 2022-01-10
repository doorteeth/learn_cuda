add_executable(printThreadInfo printThreadInfo.cu)
add_executable(sumMatrix2D-2D sumMatrix2D-2D.cu)

set_target_properties(printThreadInfo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sumMatrix2D-2D PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
