include("../test/example_models.jl");

datas = (one_fact_dat, three_mean_dat, three_path_dat)
model_funcs = (one_fact_func, three_mean_func, three_path_func)
start_values = (
    vcat(fill(1, 4), fill(0.5, 2)),
    vcat(fill(1, 9), fill(1, 3), fill(0.5, 3), fill(0.5, 6)),#, vec(mean(convert(Matrix{Float64}, three_mean_dat), dims = 1))),
    vcat(fill(1, 14), fill(0.5, 17))
    )

optimizers = (LBFGS(), GradientDescent(), Newton())


for i in 1:length(datas)
    for j in 1:length(optimizers)
        model = sem.model(model_funcs[i],
            datas[i],
            start_values[i])
    end
end

@benchmark tr = ram(test.ram(test.par)[1],
    test.ram(test.par)[2],
    test.ram(test.par)[3])

tr

@benchmark three_mean_func(tr, zeros(21))


test = sem.model(model_funcs[1], datas[1], start_values[1])

Optim.minimizer(fit(test))

Optim.minimizer(fit(test))



### model 2


test = sem.model(ram(three_mean_func(start_values)[1],
                        three_mean_func(start_values)[2],
                        three_mean_func(start_values)[3],),
                ramfunc,
                datas[2],
                start_values[2])

test.ram(start_values[2])

start_values[2]

A =     [0  :α  0
        :λ  0   9]

typeof(A[2])

ms = (test.ram(test.par))

@benchmark sem.imp_cov(test.ram(test.par))

invia = inv(I - test.ram(test.par)[3])

@benchmark ms[2]*invia*ms[1]*transpose(invia)*transpose(ms[2])

function imp_cov(D, A)
      invia = LinearAlgebra.inv(factorize((I - A)))
      #invia = convert(Array{Float64}, invia)::Array{Float64}
      imp = D[2]*invia*D[1]*transpose(invia)*transpose(D[2])
      return imp
end

function func(invia, D)
      imp = D[2]*invia*D[1]*transpose(invia)*transpose(D[2])
      return imp
end

D = three_mean_func(start_values[2])



@benchmark imp_cov(D, A)

invia = LinearAlgebra.inv!(factorize(I - D[3]))

A = convert(Array{ForwardDiff.Dual}, D[3])

@benchmark LinearAlgebra.inv!(factorize(I-D[3]))

A = test.ram(test.par)[1]
B = test.ram(test.par)[2]
C = test.ram(test.par)[3]

D = [A, B, C]

@benchmark imp_cov(D)

@benchmark three_mean_func(start_values[2])

mat[3]

A = [0  0  0  0  0  0  0  0  0  1     0     0.0
    0  0  0  0  0  0  0  0  0  0.5    0     0
    0  0  0  0  0  0  0  0  0  0.5    0     0
    0  0  0  0  0  0  0  0  0  0     1      0
    0  0  0  0  0  0  0  0  0  0     0.5    0
    0  0  0  0  0  0  0  0  0  0     0.5    0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     0.5
    0  0  0  0  0  0  0  0  0  0     0     0.5
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

@benchmark A[1,2] = 5

function tf(A::Array{Float64, 2})
    inv(I-A)
end

@benchmark tf(A)

@benchmark begin
    optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options())
end

par2 = Optim.minimizer(optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(f_tol = 1e-8)
    ))

fit(test)

par = Optim.minimizer(optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(iterations = 2)))

@benchmark optimize(
    par -> test.objective(par, test),
    test.par,
    test.optimizer,
    autodiff = :forward,
    Optim.Options(f_tol = 1e-8)
    )

par2 = Optim.minimizer(fit(test))

@code_lowered(test.objective(test.par, test))

par = Feather.read("test/comparisons/three_mean_par.feather")

test.objective(start_values[2], test)


### model 3

test = sem.model(model_funcs[3], datas[3], start_values[3];
    optimizer = LBFGS(; ))

fit(test)

test.objective(test.par, test)

sem.imp_cov(test.ram(start_values[3]))


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
