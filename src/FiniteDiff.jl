struct ram{T <: AbstractArray, N <: Union{AbstractArray, Nothing}}
    S::T
    F::T
    A::T
    M::N
end

function obj(parameters, S, F, A, M, obs_cov)
      n_man = size(model.obs.cov, 1)
      T = eltype(parameters)
      print(T)
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
      imp_cov(model.ram)
      dt = det(model.ram.imp_cov)
      #print("iteration")
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
