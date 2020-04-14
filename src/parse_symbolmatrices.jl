using ForwardDiff

struct ram{T <: AbstractArray, B <: BitArray, V <: Vector{Int64}}
    S::T
    F::T
    A::T
    M::T
    Ind_s::B
    Ind_a::B
    Ind_m::B
    parind_s::V
    parind_a::V
    parind_m::V
end


function ram(S, F, A, M, start_values)
    Ind_s = eltype.(S) .== Any
    Ind_a = eltype.(A) .== Any
    Ind_m = eltype.(M) .== Any

    sfree = sum(Ind_s)
    afree = sum(Ind_a)
    mfree = sum(Ind_m)

    parsym_a = A[Ind_a]
    parsym_s = S[Ind_s]
    parsym_m = M[Ind_m]

    parind_s = zeros(Int64, sfree)
    parind_a = zeros(Int64, afree)
    parind_m = zeros(Int64, mfree)

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

    parind_m[1] = maximum(parind_a) + 1

    for i = 2:mfree
        parind_m[i] = maximum(parind_m) + 1
        for j = 1:i
            if parsym_m[i] == parsym_m[j]
                parind_m[i] = parind_m[j]
            end
        end
    end

    S_start = copy(S)
    A_start = copy(A)
    M_start = copy(M)

    S_start[Ind_s] .= start_values[parind_s]
    A_start[Ind_a] .= start_values[parind_a]
    M_start[Ind_m] .= start_values[parind_m]

    return ram(
        convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, S_start),
        convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, F),
        convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, A_start),
        convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, M_start),
        Ind_s, Ind_a, Ind_m, parind_s, parind_a, parind_m)
end

function (ram::ram)(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
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

myram = ram(S, F, A, M, start_values)

new_values = [94 32 32 41 52 63 73.0]

optim_parameters = convert(Array{ForwardDiff.Dual}, new_values)

@benchmark myram(optim_parameters)

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
