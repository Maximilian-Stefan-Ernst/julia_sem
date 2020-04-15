using sem, Test, Feather, BenchmarkTools, Distributions, Optim
# using Test, Feather, BenchmarkTools, Distributions, Optim
### Modelle
one_fact_dat = Feather.read("test/comparisons/one_fact_dat.feather")
one_fact_par = Feather.read("test/comparisons/one_fact_par.feather")

function one_fact_func(x)#::Union{NTuple(3, Array{Float64}), NTuple(3, AbstractArray)}
    S = [x[1] 0 0 0
        0 x[2] 0 0
        0 0 x[3] 0
        0 0 0 x[4]]

    F = [1 0 0 0
        0 1 0 0
        0 0 1 0]

    A = [0 0 0 1
        0 0 0 x[5]
        0 0 0 x[6]
        0 0 0 0]

    return [S, F, A]
end

three_mean_dat = Feather.read("test/comparisons/three_mean_dat.feather")
three_mean_par = Feather.read("test/comparisons/three_mean_par.feather")

function three_mean_func(x)
    S =[x[1] 0 0 0 0 0 0 0 0 0 0 0.0
        0 x[2] 0 0 0 0 0 0 0 0 0 0
        0 0 x[3] 0 0 0 0 0 0 0 0 0
        0 0 0 x[4] 0 0 0 0 0 0 0 0
        0 0 0 0 x[5] 0 0 0 0 0 0 0
        0 0 0 0 0 x[6] 0 0 0 0 0 0
        0 0 0 0 0 0 x[7] 0 0 0 0 0
        0 0 0 0 0 0 0 x[8] 0 0 0 0
        0 0 0 0 0 0 0 0 x[9] 0 0 0
        0 0 0 0 0 0 0 0 0    x[10] x[13] x[14]
        0 0 0 0 0 0 0 0 0    x[13] x[11] x[15]
        0 0 0 0 0 0 0 0 0    x[14] x[15] x[12]]

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
        0  0  0  0  0  0  0  0  0  x[16] 0     0
        0  0  0  0  0  0  0  0  0  x[17] 0     0
        0  0  0  0  0  0  0  0  0  0     1     0
        0  0  0  0  0  0  0  0  0  0     x[18] 0
        0  0  0  0  0  0  0  0  0  0     x[19] 0
        0  0  0  0  0  0  0  0  0  0     0     1
        0  0  0  0  0  0  0  0  0  0     0     x[20]
        0  0  0  0  0  0  0  0  0  0     0     x[21]
        0  0  0  0  0  0  0  0  0  0     0     0
        0  0  0  0  0  0  0  0  0  0     0     0
        0  0  0  0  0  0  0  0  0  0     0     0]

    M = [x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], 0, 0, 0]

    return [S, F, A, M]
end

function ramfunc(ram::ram, par)
    for i = 1:12 ram.S[i,i] = par[i] end
    ram.S[11, 10] = par[13]
    ram.S[10, 11] = par[13]
    ram.S[12, 10] = par[14]
    ram.S[10, 12] = par[14]
    ram.S[11, 12] = par[15]
    ram.S[12, 11] = par[15]

    ram.A[2, 10] = par[16]
    ram.A[3, 10] = par[17]
    ram.A[5, 11] = par[18]
    ram.A[6, 11] = par[19]
    ram.A[8, 12] = par[20]
    ram.A[9, 12] = par[21]
end

three_path_dat = Feather.read("test/comparisons/three_path_dat.feather")
three_path_par = Feather.read("test/comparisons/three_path_par.feather")
lav_start_par = Feather.read("test/comparisons/start_par.feather")

function three_path_func(x)

  S =[x[1]  0     0     0     0     0     0     0     0     0     0     0     0     0
      0     x[2]  0     0     0     0     0     0     0     0     0     0     0     0
      0     0     x[3]  0     0     0     0     0     0     0     0     0     0     0
      0     0     0     x[4]  0     0     0     x[15] 0     0     0     0     0     0
      0     0     0     0     x[5]  0     x[16] 0     x[17] 0     0     0     0     0
      0     0     0     0     0     x[6]  0     0     0     x[18] 0     0     0     0
      0     0     0     0     x[16] 0     x[7]  0     0     0     x[19] 0     0     0
      0     0     0     x[15] 0     0     0     x[8]  0     0     0     0     0     0
      0     0     0     0     x[17] 0     0     0     x[9]  0     x[20] 0     0     0
      0     0     0     0     0     x[18] 0     0     0     x[10] 0     0     0     0
      0     0     0     0     0     0     x[19] 0     x[20] 0     x[11] 0     0     0
      0     0     0     0     0     0     0     0     0     0     0     x[12] 0     0
      0     0     0     0     0     0     0     0     0     0     0     0     x[13] 0
      0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

    F =[1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]

  A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
      0  0  0  0  0  0  0  0  0  0  0     x[21] 0     0
      0  0  0  0  0  0  0  0  0  0  0     x[22] 0     0
      0  0  0  0  0  0  0  0  0  0  0     0     1     0
      0  0  0  0  0  0  0  0  0  0  0     0     x[23] 0
      0  0  0  0  0  0  0  0  0  0  0     0     x[24] 0
      0  0  0  0  0  0  0  0  0  0  0     0     x[25] 0
      0  0  0  0  0  0  0  0  0  0  0     0     0     1
      0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
      0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
      0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
      0  0  0  0  0  0  0  0  0  0  0     0     0     0
      0  0  0  0  0  0  0  0  0  0  0     x[29] 0     0
      0  0  0  0  0  0  0  0  0  0  0     x[30] x[31] 0]

    return [S, F, A]
end

function growth_func(x)

  S =[x[1]  0  0  0  0  0  0  0  0  0  0  0
      0  x[2]  0  0  0  0  0  0  0  0  0  0
      0  0  x[3]  0  0  0  0  0  0  0  0  0
      0  0  0  x[4]  0  0  0  0  0  0  0  0
      0  0  0  0  x[5]  0  0  0  0  0  0  0
      0  0  0  0  0  x[6]  0  0  0  0  0  0
      0  0  0  0  0  0  x[7]  0  0  0  0  0
      0  0  0  0  0  0  0  x[8]  0  0  0  0
      0  0  0  0  0  0  0  0  x[9]  0  0  0
      0  0  0  0  0  0  0  0  0  x[10]  0  0
      0  0  0  0  0  0  0  0  0  0  x[11]  x[13]
      0  0  0  0  0  0  0  0  0  0  x[13]  x[12]]


    F =[1 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0]

  A =[0  0  0  0  x[14] 0     0     0     0     0     1     0
      0  0  0  0  0     x[15] 0     0     0     0     1     1
      0  0  0  0  0     0     x[16] 0     0     0     1     2
      0  0  0  0  0     0     0     x[17] 0     0     1     3
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     0     0     0     0
      0  0  0  0  0     0     0     0     x[18] x[20] 0     0
      0  0  0  0  0     0     0     0     x[19] x[21] 0     0]

    return (S, F, A)
end
