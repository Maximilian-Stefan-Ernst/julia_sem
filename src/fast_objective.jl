function (objective::SemML)(parameters, model::model)
      #obs_cov = model.obs.cov
      n_man = size(obs_cov, 1)
      model.ram(parameters)
      #if rank(I-matrices[3]) < size(matrices[3], 1)
      #      return 100.0
      #end
      #model.ramf(model.ram, parameters)
      imp_cov(model.ram)
      dt = det(model.ram.imp_cov)
      if dt < 0.0
            return 100000.0
      end
      F_ML = log(dt) + tr(obs_cov*inv(imp_cov)) - log(det(obs_cov)) - n_man
      if !isa(model.ram.M, Nothing)
          mean_diff = model.obs.mean - sem.imp_mean(matrices)
          F_ML = F_ML + transpose(mean_diff)*inv(imp_cov)*mean_diff
      end
      return F_ML
end

function imp_cov(ram::ram)
    invia = Array{Float64}(undef, 12, 12)
    invia .= LinearAlgebra.inv!(factorize(I - ram.A))
    ram.imp_cov .= ram.F*invia*ram.S*transpose(invia)*transpose(ram.F)
end
