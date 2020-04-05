# Maximum Likelihood Estimation

function ML_wrap(parameters, model::model)
      matrices = model.ram(parameters)
      ML(matrices, model)
end

# standard
function ML(parameters, matrices::NTuple{3, Any}, model::model, reg::Nothing)
      obs_cov = model.obs_cov
      n_man = size(obs_cov, 1)
      Cov_Exp = imp_cov(matrices)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end

# with meanstructure
function ML(parameters, matrices::NTuple{4, Any}, model::model, reg::Nothing)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      n_man = size(obs_cov, 1)
      Cov_Exp = imp_cov(matrices)
      F_ML = log(det(Cov_Exp)) + tr(inv(Cov_Exp)*obs_cov) +
                  transpose(obs_mean - Mean_Exp)*inv(Cov_Exp)*
                        (obs_mean - Mean_Exp)
      return F_ML
end

### RegSem
# Ridge
function ML(parameters, matrices::NTuple{3, Any}, model::model, reg::reg)
      # get data
      obs_cov = model.obs_cov
      #
      n_man = size(obs_cov, 1)
      # compute implied cov + mean
      Cov_Exp = imp_cov(matrices)
      # compute F-Value
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man +
                  penalty_term(parameters, reg)
      return F_ML
end

function ML(parameters, matrices::NTuple{4, Any}, model::model, reg::reg)
      # get data
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      #
      n_man = size(obs_cov, 1)
      # compute implied cov + mean
      Cov_Exp = imp_cov(matrices)
      Mean_Exp = matrices[2]*inv(I-matrices[3])*matrices[4]
      # compute F-Value
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man +
                  transpose(obs_mean - Mean_Exp)*inv(Cov_Exp)*
                        (obs_mean - Mean_Exp) +
                  penalty_term(parameters, reg)
      return F_ML
end

function penalty_term(parameters, reg::reg)
      lasso = reg.lasso
      lasso_pen = reg.lasso_pen
      ridge = reg.ridge
      ridge_pen = reg.ridge_pen
      pen = lasso_pen*sum(transpose(parameters)[lasso]) +
      ridge_pen*sum(transpose(parameters)[ridge].^2)
end

function penalty_term(parameters, reg::reg{Array{Bool,2},Float64,Nothing,Nothing})
      lasso = reg.lasso
      lasso_pen = reg.lasso_pen
      pen = lasso_pen*sum(transpose(parameters)[lasso])
end

function penalty_term(parameters, reg::reg{Nothing,Nothing,Array{Bool,2},Float64})
      ridge = reg.ridge
      ridge_pen = reg.ridge_pen
      ridge_pen*sum(transpose(parameters)[ridge].^2)
end
# FIML
### to add


### test - takes only relevant fields instead of the whole model object
function ML_test(parameters, ram, obs_cov)
      n_man = size(obs_cov, 1)
      matrices = ram(parameters)
      Cov_Exp = matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end
