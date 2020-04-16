using sem, Test, Feather, BenchmarkTools, Distributions, Optim, ForwardDiff
# using Test, Feather, BenchmarkTools, Distributions, Optim
### Modelle
one_fact_dat = Feather.read("test/comparisons/one_fact_dat.feather")
one_fact_par = Feather.read("test/comparisons/one_fact_par.feather")


    S =[:x1 0 0 0
        0 :x2 0 0
        0 0 :x3 0
        0 0 0 :x4]

    F =[1 0 0 0
        0 1 0 0
        0 0 1 0]

    A =[0 0 0 1
        0 0 0 :λ₂
        0 0 0 :λ₃
        0 0 0 0]

one_fact_ram = ram(S, F, A, nothing, [1.0 1 1 1 0.5 0.5],
    zeros(3,3))


three_mean_dat = Feather.read("test/comparisons/three_mean_dat.feather")
three_mean_par = Feather.read("test/comparisons/three_mean_par.feather")

S =[:x1 0 0 0 0 0 0 0 0 0 0 0.0
    0 :x2 0 0 0 0 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0 0 0 0 0
    0 0 0 :x4 0 0 0 0 0 0 0 0
    0 0 0 0 :x5 0 0 0 0 0 0 0
    0 0 0 0 0 :x6 0 0 0 0 0 0
    0 0 0 0 0 0 :x7 0 0 0 0 0
    0 0 0 0 0 0 0 :x8 0 0 0 0
    0 0 0 0 0 0 0 0 :x9 0 0 0
    0 0 0 0 0 0 0 0 0 :x10 :x13 :x14
    0 0 0 0 0 0 0 0 0 :x13 :x11 :x15
    0 0 0 0 0 0 0 0 0 :x14 :x15 :x12]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  1     0     0.0
    0  0  0  0  0  0  0  0  0  :x16 0     0
    0  0  0  0  0  0  0  0  0  :x17 0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     :x18 0
    0  0  0  0  0  0  0  0  0  0     :x19 0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     :x20
    0  0  0  0  0  0  0  0  0  0     0     :x21
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

M = [:x22, :x23, :x24, :x25, :x26, :x27, :x28, :x29, :x30, 0, 0, 0]

start_val = vcat(fill(1, 10), [0.5, 0.5, 1, 0.5, 1], fill(0.5, 6),
    vec(mean(convert(Matrix{Float64}, three_mean_dat), dims = 1)))

three_mean_ram = ram(S, F, A, M, start_val,
    zeros(9,9))



three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")
lav_start_par = Feather.read("test/comparisons/start_par.feather")
