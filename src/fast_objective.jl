struct SemML <: SemObjective end

function (objective::SemML)(parameters, model::model)
      n_man = size(model.obs.cov, 1)
      T = eltype(parameters)
      #print(T)
      if isa(model.ram.A[1,1], T)
      else
            model.ram.S = convert(Array{T}, model.ram.S)
            model.ram.A = convert(Array{T}, model.ram.A)
            model.ram.imp_cov = convert(Array{T}, model.ram.imp_cov)
            if isa(model.ram.M, Nothing)
            else
                  model.ram.M = convert(Array{T}, model.ram.M)
            end
            print("conversion took place")
      end
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
      F_ML = log(dt) + tr(model.obs.cov*inv(model.ram.imp_cov)) - log(det(model.obs.cov)) - n_man
      if !isa(model.ram.M, Nothing)
          mean_diff = model.obs.mean - imp_mean(model.ram)
          F_ML = F_ML + transpose(mean_diff)*inv(model.ram.imp_cov)*mean_diff
      end
      return F_ML
end

function imp_cov(ram::ram)
    #invia = similar(ram.A)
    invia = inv(I - ram.A)
    #invia .= LinearAlgebra.inv!(factorize(I - ram.A))
    ram.imp_cov .= ram.F*invia*ram.S*transpose(invia)*transpose(ram.F)
end

function imp_mean(ram::ram)
      ram.F*inv(I - ram.A)*ram.M
      #ram.F*LinearAlgebra.inv!(factorize(I - ram.A))*ram.M
end
