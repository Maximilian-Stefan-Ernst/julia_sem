using sem, Test, Feather, BenchmarkTools, Distributions,
        Optim

function holz_onef_mod(x)::Tuple{Any, Any, Any}
    S = [x[1] 0 0 0
        0 x[2] 0 0
        0 0 x[3] 0
        0 0 0 x[4]]

    F = [1.0 0 0 0
        0 1 0 0
        0 0 1 0]

    A = [0 0 0 1
        0 0 0 x[5]
        0 0 0 x[6]
        0 0 0 0]

    return (S, F, A)
end

function holz_onef_mod_mean(x)
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

    M = [0
        x[8]
        x[9]
        x[7]]

    return (S, F, A, M)
end

using Feather

holz_onef_dat = Feather.read("test/comparisons/holz_onef_dat.feather")
holz_onef_par = Feather.read("test/comparisons/holz_onef_par.feather")

mymod_lbfgs =
    sem_model(holz_onef_mod,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    data = convert(Matrix{Float64}, holz_onef_dat))


mymod_lbfgs =
    sem_model(holz_onef_mod_mean,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 5.0, 1.0, -3.0];
    data = convert(Matrix{Float64}, holz_onef_dat))


sem_fit!(mymod_lbfgs)

mymod_lbfgs

mymod = model(mymod_lbfgs)

objective = parameters ->
        mymod_lbfgs.est(parameters, mymod)

result = optimize(objective, mymod.par, LBFGS(),
                autodiff = :forward)

mymod_lbfgs

###

@benchmark begin
    mymod_lbfgs =
        sem_model(holz_onef_mod,
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
        data = convert(Matrix{Float64}, holz_onef_dat))
    sem_fit!(mymod_lbfgs)
end

@benchmark begin
    mymod_lbfgs =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    opt = "test")
    sem_fit!(mymod_lbfgs)
end

@benchmark begin
    mymod_lbfgs =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    obs_cov = Distributions.cov(convert(Matrix{Float64}, holz_onef_dat)),
    obs_mean = mean(convert(Matrix{Float64}, holz_onef_dat), dims = 1),
    est = sem.ML)
    objective = parameters ->
            ML2(parameters, mymod_lbfgs)
    result =
            optimize(objective, mymod_lbfgs.par, LBFGS(),
            autodiff = :forward)
end


### type stability
mymod_lbfgs =
    sem(holz_onef_mod,
    convert(Matrix{Float64}, rand(100, 100)),
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    obs_cov = Distributions.cov(convert(Matrix{Float64}, holz_onef_dat)),
    obs_mean = mean(convert(Matrix{Float64}, holz_onef_dat), dims = 1))

setfield!(mymod_lbfgs, :logl, 5.0)

mymod_lbfgs

mymod_lbfgs = @set mymod_lbfgs.logl = 1.0

const B = rand(30, 1000000)

mymod_lbfgs = @set mymod_lbfgs.data = B

f(mymod_lbfgs)

function f(mymod_lbfgs)
    mymod_lbfgs = @set mymod_lbfgs.logl = nothing
end

@benchmark my = mymod_lbfgs

set(mymod_lbfgs, lens, rand(10, 10))

mymod_lbfgs

mymod_lbfgs = set(mymod_lbfgs, lens, nothing)

@benchmark begin
    f(mymod_lbfgs)
end

@benchmark setfield!(mymod_lbfgs, :data, B)

function ML3(parameters::Union{Array{Float64}, AbstractArray}, model::model)
      obs_cov = model.obs_cov
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end

@code_warntype holz_onef_mod(ones(6))
@code_warntype mymod_lbfgs.ram(ones(6))
@code_warntype ML3(ones(6), model(mymod_lbfgs))

@benchmark begin
    mymod_lbfgs =
        model(holz_onef_mod, holz_onef_dat,
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
        obs_cov = Distributions.cov(convert(Matrix{Float64}, holz_onef_dat)),
        obs_mean = mean(convert(Matrix{Float64}, holz_onef_dat), dims = 1),
        est = sem.ML)
        objective = parameters ->
            ML3(parameters, mymod_lbfgs)
        result =
            optimize(objective, mymod_lbfgs.par, LBFGS(),
            autodiff = :forward)
end

@code_warntype optimize(objective, mymod_lbfgs.par, LBFGS(),
    autodiff = :forward)

mymod_newton =
    model(holz_onef_mod, holz_onef_dat,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    opt = "Newton")
sem_fit!(mymod_newton)


### testarea

struct mystruc{T <: Function}
    a::T
    mystruc{T}(a) where {T <: Function} = new(a)
end


function mysum(x, y)
    x + y
end

mystruc(func::T) where {T <: Function} = mystruc{T}(func)

teststruc = mystruc(mysum)

function wrapper(struc, y, x)
    res = struc.a(y, x)
end

@code_warntype mysum(5.0, 5.0)

@code_warntype teststruc.a(5.0, 5.0)

@code_warntype wrapper(teststruc, 5.0, 5.0)

ram([1.0], [1.0], [1.0])
