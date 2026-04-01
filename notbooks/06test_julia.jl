# 06test_julia.jl
# Julia Script

#=
Description: 
Author: KIST
Date: 26. 3. 26.
=#
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "python"
using PythonCall


np = pyimport("numpy")

println("==== 1. 파이썬 원본 배열 생성 ====")
py_array = np.zeros((3, 3))
println(py_array)

jl_array = PyArray(py_array)

jl_array[1, 1] = 99.0
jl_array[2, 3] = 42.0

println("\n==== 2. 줄리아에서 수정한 배열 ====")
println(jl_array)

println("\n==== 3. 파이썬 객체 다시 확인 (Zero-copy 증명!) ====")
println(py_array)
