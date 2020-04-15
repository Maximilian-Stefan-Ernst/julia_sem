using ForwardDiff, Random

struct ram{T <: AbstractArray, N <: Union{AbstractArray, Nothing},
    B <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
    V2 <: Vector{Any}}
    S::T
    F::T
    A::T
    M::N
    Ind_s::B
    Ind_a::B
    Ind_m::B
    parind_s::V
    parind_a::V
    parind_m::V
    imp_cov::IM
    names::V2
end


function ram(S, F, A, M, start_values, imp_cov)
    Ind_s = eltype.(S) .== Any
    Ind_a = eltype.(A) .== Any

    sfree = sum(Ind_s)
    afree = sum(Ind_a)

    parsym_a = A[Ind_a]
    parsym_s = S[Ind_s]

    parind_s = zeros(Int64, sfree)
    parind_a = zeros(Int64, afree)

    parind_s[1] = 1

    for i = 2:sfree
        parind_s[i] = maximum(parind_s) + 1
        for j = 1:i
            if parsym_s[i] == parsym_s[j]
                parind_s[i] = parind_s[j]
            end
        end
    end

    parind_a[1] = maximum(parind_s) + 1

    for i = 2:afree
        parind_a[i] = maximum(parind_a) + 1
        for j = 1:i
            if parsym_a[i] == parsym_a[j]
                parind_a[i] = parind_a[j]
            end
        end
    end

    if !isa(M, Nothing)
        Ind_m = eltype.(M) .== Any
        mfree = sum(Ind_m)
        parsym_m = M[Ind_m]
        parind_m = zeros(Int64, mfree)


        parind_m[1] = maximum(parind_a) + 1

        for i = 2:mfree
            parind_m[i] = maximum(parind_m) + 1
            for j = 1:i
                if parsym_m[i] == parsym_m[j]
                    parind_m[i] = parind_m[j]
                end
            end
        end

        M_start = copy(M)
        M_start[Ind_m] .= start_values[parind_m]
        M_start = convert(Array{Float64}, M_start)
    else
        M_start = nothing
        Ind_m = bitrand(2,2)
        parind_m = rand(Int64, 5)
    end

    S_start = copy(S)
    A_start = copy(A)


    S_start[Ind_s] .= start_values[parind_s]
    A_start[Ind_a] .= start_values[parind_a]


    return ram(
        convert(Array{Float64}, S_start),
        convert(Array{Float64}, F),
        convert(Array{Float64}, A_start),
        M_start,
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, S_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, F),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, A_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, M_start),
        Ind_s, Ind_a, Ind_m, parind_s, parind_a, parind_m, imp_cov,
        [unique(parsym_s); unique(parsym_a)])
end

function (ram::ram{Array{Float64,2},Array{Float64,2}})(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram{Array{Float64,2},Nothing})(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
end

### testmodel

S =[:v₁ 0   0   0
    0   :v₂ 0   0
    0   0   :v₂ 0
    0   0   0   :v₃]

F = [1 0 0 0
    0 1 0 0
    0 0 1 0]

A = [0 0 0 1
    0 0 0 :λ₁
    0 0 0 :λ₁
    0 0 0 :λ₂
    0 0 0 0]

M = [0 :m₁ :m₁ :m₂]

start_values = [10.0 20 30 40 50 60 70]

myram = ram(S, F, A, nothing, start_values)

new_values = [94 32 32 41 52 63 73.0]

optim_parameters = convert(Array{ForwardDiff.Dual}, new_values)

myram(new_values)

myram


### testing area

arr = convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, rand(10, 10))

par = [ForwardDiff.Dual(1.09) ForwardDiff.Dual(2.1) ForwardDiff.Dual(1.4) ForwardDiff.Dual(1.7)]

Ind = BitArray(zeros(10, 10))

indic = rand(1:(10*10), 5)

Ind[indic] .= 1

alloc = Vector{ForwardDiff.Dual{Nothing,Float64,0}}(undef, sum(Ind))

Ind_par = rand(1:4, sum(Ind))

alloc .= par[Ind_par]

function tf(A, Ind, alloc)
    A[Ind] .= alloc
end

function tf2(A, Ind, par, Ind_par)
    A[Ind] .= par[Ind_par]
end

@benchmark tf(arr, Ind, alloc)

@benchmark tf2(arr, Ind, par, Ind_par)



### implied covariance

S =[:x1 0 0 0 0 0 0 0 0 0 0 0.0
    0 :x2 0 0 0 0 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0 0 0 0 0
    0 0 0 :x4 0 0 0 0 0 0 0 0
    0 0 0 0 :x5 0 0 0 0 0 0 0
    0 0 0 0 0 :x6 0 0 0 0 0 0
    0 0 0 0 0 0 :x7 0 0 0 0 0
    0 0 0 0 0 0 0 :x8 0 0 0 0
    0 0 0 0 0 0 0 0 :x9 0 0 0
    0 0 0 0 0 0 0 0 0 :x10 :x13 :x14
    0 0 0 0 0 0 0 0 0 :x13 :x11 :x15
    0 0 0 0 0 0 0 0 0 :x14 :x15 :x12]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  1     0     0.0
    0  0  0  0  0  0  0  0  0  :x16 0     0
    0  0  0  0  0  0  0  0  0  :x17 0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     :x18 0
    0  0  0  0  0  0  0  0  0  0     :x19 0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     :x20
    0  0  0  0  0  0  0  0  0  0     0     :x21
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

start_values = vcat(fill(1, 10), [0.5, 0.5, 1, 0.5, 1], fill(0.5, 6))

myram = ram(S, F, A, nothing, start_values, Array{Float64}(undef, 9, 9))

invia = Array{Float64}(undef, 12, 12)

amul = Array{Float64}(undef, 9, 12)
bmul = Array{Float64}(undef, 9, 12)
cmul = Array{Float64}(undef, 9, 12)

function imp_cov(ram::ram)
    invia = Array{Float64}(undef, 12, 12)
    invia .= LinearAlgebra.inv!(factorize(I - ram.A)) # invers of I(dentity) minus A matrix
    mul!(amul, ram.F, invia)
    mul!(bmul, amul, ram.S)
    mul!(cmul, bmul, transpose(invia))
    mul!(ram.imp_cov, cmul, transpose(ram.F))
end

function imp_cov(ram::ram)
    invia = Array{Float64}(undef, 12, 12)
    invia .= LinearAlgebra.inv!(factorize(I - ram.A))
    ram.imp_cov .= ram.F*invia*ram.S*transpose(invia)*transpose(ram.F)
end

@benchmark imp_cov(myram, amul, bmul, cmul)

function obfun(parameters, model::model)
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

myram.A

@time LinearAlgebra.inv!(factorize(myram.imp_cov))

@time LinearAlgebra.LAPACK.potri!('U', myram.imp_cov)

function imp_cov(matrices)
      invia = inv(I - matrices[3]) # invers of I(dentity) minus A matrix
      imp = matrices[2]*invia*matrices[1]*transpose(invia)*transpose(matrices[2])
      return imp
end

imp_cov(three_mean_func(vcat(fill(1, 9), fill(1, 3), fill(0.5, 3), fill(0.5, 6), fill(0, 9))))

matr = three_mean_func(vcat(fill(1, 9), fill(1, 3), fill(0.5, 3), fill(0.5, 6), fill(0, 9)))

matr[2] == myram.F
