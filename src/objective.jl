# Maximum Likelihood Estimation

function ML_wrap(parameters, model::model)
      matrices = model.ram(parameters)
      ML(matrices, model)
end

function ML(matrices::NTuple{3, Any}, model::model)
      obs_cov = model.obs_cov
      n_man = size(obs_cov, 1)
      Cov_Exp = imp_cov(matrices)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      return F_ML
end

function ML(matrices::NTuple{4, Any}, model::model)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      n_man = size(obs_cov, 1)
      Cov_Exp = imp_cov(matrices)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) - log(det(obs_cov)) - n_man
      Mean_Exp = matrices[2]*inv(I-matrices[3])*matrices[4]
      F_ML = log(det(Cov_Exp)) + tr(inv(Cov_Exp)*obs_cov) +
                  transpose(obs_mean - Mean_Exp)*inv(Cov_Exp)*
                        (obs_mean - Mean_Exp)
      return F_ML
end

### RegSem
function ML_lasso(parameters, model)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      reg_vec = model.rec_vec
      penalty = model.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = imp_cov(model, parameters)
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec])
      return F_ML
end

function ML_ridge(parameters; ram, obs_cov, reg_vec, penalty)
      obs_cov = model.obs_cov
      obs_mean = model.obs_mean
      reg_vec = model.rec_vec
      penalty = model.penalty
      n_man = size(obs_cov, 1)
      matrices = model.ram(parameters)
      Cov_Exp = matrices[2]*inv(I-matrices[3])*matrices[1]*transpose(inv(I-matrices[3]))*transpose(matrices[2])
      F_ML = log(det(Cov_Exp)) + tr(obs_cov*inv(Cov_Exp)) -
                  log(det(obs_cov)) - n_man + penalty*sum(transpose(parameters)[reg_vec].^2)
      return F_ML
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
