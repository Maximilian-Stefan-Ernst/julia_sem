import StatsBase.fit
function fit(model::model)
    optimize(
    par -> model.objective(par, model),
    model.par,
    model.optimizer,
    autodiff = :forward,
    Optim.Options(x_tol = 1e-6,
                    f_tol = 1e-6))
end
