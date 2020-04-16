include("../test/example_models0.jl");

using BenchmarkTools, Test

datas = (one_fact_dat, three_mean_dat, three_path_dat)

start_values = (
    vcat(fill(1, 4), fill(0.5, 2)),
    vcat(fill(1, 9), fill(1, 3), fill(0.5, 3), fill(0.5, 6), vec(mean(convert(Matrix{Float64}, three_mean_dat), dims = 1))),
    vcat(fill(1.0, 11), fill(0.05, 3), fill(0.0, 6), fill(1.0, 8), fill(0, 3))
    )


### model 1

test = sem.model(one_fact_ram, datas[1], start_values[1])

@benchmark fit(test)

# compare parameters
pars = Optim.minimizer(fit(test))
par_order = [collect(4:7); collect(2:3)]

@test all(abs.(pars .- one_fact_par.est[par_order]) .< 0.02)


### model 2
three_mean_ram = ram(S, F, A, M, start_val,
    zeros(9,9))

test = sem.model(three_mean_ram,
                datas[2],
                start_values[2])


@benchmark fit(test)

pars = Optim.minimizer(fit(test))''
par_order = [collect(19:33); 2;3;5;6;8;9; collect(10:18)]

@test all(abs.(pars .- three_mean_par.est[par_order]) .< 0.02)

@benchmark test.objective(convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, test.par),
    test)

### model 3

using LinearAlgebra

test = sem.model(fake_ram,
                model_funcs[3],
                datas[3],
                start_values[3])

#start_values[3][collect(1:11)] .= diag(test.obs.cov)

#start_values[3][collect(12:14)] .= 0.0

#pars = Optim.minimizer(fit(test))

par_order = [collect(21:34);  collect(15:20); 2;3; 5;6;7; collect(9:14)]

@test all(abs.(pars .- three_path_par.est[par_order]) .< 0.02*abs.(pars))



test.objective(three_path_par.est[par_order], test)

lav_start_par.est[par_order]

using LineSearches

optimize(
    par -> test.objective(par, test),
    lav_start_par.est[par_order],
    LBFGS(),
    autodiff = :forward,
    Optim.Options(allow_f_increases = false#,
                    #x_tol = 1e-4,
                    #f_tol = 1e-4
                    ))

function func(invia, D)
      imp = D[2]*invia*D[1]*transpose(invia)*transpose(D[2])
      return imp
end



par = Feather.read("test/comparisons/three_mean_par.feather")

test.objective(start_values[2], test)

### minimal working example
using Optim, ForwardDiff

### old solution

function mymatrix(par)
    A =    [0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 par[1]]
    return A
end

function old_objective(par, func)
    A = func(par)
    return A[4,8]^2
end

@benchmark optimize(par -> old_objective(par, mymatrix),
            [5.0],
            LBFGS(),
            autodiff = :forward)

### gives an error

mutable struct MyStruct{T}
    a::T
end

A =  [0.0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0]



teststruct = MyStruct(A)

function prepare_struct(par, mystruct)
    mystruct.a[4,8] = par[1]
    return mystruct
end

function objective(mystruct)
    mystruct.a[4,8]^2
end

function wrap(par, mystruct)
    mystruct = prepare_struct(par, mystruct)
    return objective(mystruct)
end

optimize(par -> wrap(par, teststruct),
            [5.0],
            LBFGS(),
            autodiff = :forward)

### working solution

function myconvert(par, mystruct)
    T = eltype(par)
    new_array = convert(Array{T}, mystruct.a)
    mystruct_conv = MyStruct(new_array)
    return mystruct_conv
end

function new_wrap(par, mystruct)
    mystruct = myconvert(par, mystruct)
    mystruct = prepare_struct(par, mystruct)
    return objective(mystruct)
end

@benchmark optimize(par -> new_wrap(par, teststruct),
            [5.0],
            LBFGS(),
            autodiff = :forward)

### solutin with diffeq
using Optim, OrdinaryDiffEq, LinearAlgebra
using DiffEqBase, ForwardDiff

mutable struct MyStruct{T}
    a::T
end

A =  [0.0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0]

teststruct = MyStruct(A)

function wrap(par, A, matr)
    matr = DiffEqBase.get_tmp(matr, A)
    matr = one(eltype(par))*A
    return matr
    #matr[4,8] = par[1]
    #return matr[4,8]^2
end

@benchmark wrap(ForwardDiff.Dual(8.0), ones(5,5),
                DiffEqBase.dualcache(zeros(5,5)))

optimize(par -> wrap(par, DiffEqBase.dualcache(teststruct.a)),
            [5.0],
            LBFGS(),
            autodiff = :forward)


function foo(du, u, (A, tmp))
    tmp = DiffEqBase.get_tmp(tmp, u)
    mul!(tmp, A, u)
    @. du = u + tmp
    nothing
end

foo(ones(5, 5), (0., 1.0),
        (ones(5,5), DiffEqBase.dualcache(zeros(5,5))))
solve(prob, TRBDF2())


A = rand(10, 10)

par = zeros(3)
