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

myreg = reg(
    lasso = [false false false false true true],
    lasso_pen = 0.01
)

myreg = reg(
    ridge = [false false false false true true false false false],
    ridge_pen = 0.01
)

myreg = reg(
    lasso = [false false false false true true],
    lasso_pen = 0.01,
    ridge = [false false false false true true],
    ridge_pen = 0.01
)

mymod_lbfgs =
    sem_model(holz_onef_mod,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
    data = convert(Matrix{Float64}, holz_onef_dat),
    reg = myreg)

mymod_lbfgs =
    sem_model(holz_onef_mod_mean,
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 5.0, 1.0, -3.0];
    data = convert(Matrix{Float64}, holz_onef_dat),
    reg = myreg)


sem_fit!(mymod_lbfgs)

mymod_lbfgs

mymod = model(mymod_lbfgs)

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
        sem_model(holz_onef_mod,
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0];
        data = convert(Matrix{Float64}, holz_onef_dat),
        reg = myreg)
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
