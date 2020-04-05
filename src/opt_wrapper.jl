# wrapper to call the optimizer
function opt_sem(model)
      if model.opt == "LBFGS"
            objective = parameters ->
                  model.est(parameters, model.ram(parameters), model, model.reg)
            result =
                  optimize(objective, model.par, LBFGS(),
                        autodiff = :forward)
      elseif model.opt == "test"
            objective = parameters ->
                  ML_test(parameters, model.ram, model.obs_cov)
            result =
                  optimize(objective, model.par, LBFGS(),
                        autodiff = :forward)
      elseif model.opt == "Newton"
            objective = TwiceDifferentiable(
                  parameters -> model.est(parameters, model),
                        model.par,
                        autodiff = :forward)
                  result = optimize(objective, model.par)
      else
            error("Unknown Optimizer")
      end
end


lamb = x -> sqrt(sum(x, dims = 2))

lamb([5.0 5.0])
