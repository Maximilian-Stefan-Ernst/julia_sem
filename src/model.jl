using Random

abstract type SemObjective end

mutable struct ram{T <: AbstractArray, N <: Union{AbstractArray, Nothing},
    B1 <: BitArray, B2 <: BitArray, V <: Vector{Int64}, #IM <: AbstractArray,
    V2 <: Vector{Any}}
    S
    F::T
    A
    MS::N
    M
    Ind_s::B1
    Ind_a::B1
    Ind_m::B2
    parind_s::V
    parind_a::V
    parind_m::V
    imp_cov
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
        #M_start = convert(Array{Float64}, M_start)
        M_start = convert(
            Array{Float64}, M_start)
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
    #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, S_start),
    #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, F),
    #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, A_start),
        convert(Array{Float64}, S_start),
        convert(Array{Float64}, F),
        convert(Array{Float64}, A_start),
        M_start, M_start,
        Ind_s, Ind_a, Ind_m, parind_s, parind_a, parind_m, imp_cov,
        [unique(parsym_s); unique(parsym_a)])
end

function (ram::ram{Array{Float64,2},Array{Float64,1}})(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram{Array{Float64,2},Nothing})(parameters)
    #T = eltype(parameters)
    #ram.S = convert(Array{T}, ram.S)
    #ram.A = convert(Array{T}, ram.A)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
end

mutable struct model{
        RAM <: ram,
        OBS,
        PAR <: AbstractVecOrMat,
        OBJ <: SemObjective,
        OPT <: Optim.AbstractOptimizer}
    ram::RAM
    obs::OBS
    par::PAR
    objective::OBJ
    optimizer::OPT
    imp_cov
    optimizer_result
    par_uncertainty
    fitmeasure
end

struct SemObs{D, C, M}
    data::D
    cov::C
    mean::M
end

struct SemCalcCov end
struct SemCalcMean end

function SemObs(data; cov = SemCalcCov(), mean = SemCalcMean())
    SemObs(data, cov, mean)
end
import DataFrames.DataFrame
function SemObs(data::DataFrame; cov = SemCalcCov(), mean = SemCalcMean())
    data = convert(Matrix, data)
    SemObs(data, cov, mean)
end

function SemObs(data, cov::SemCalcCov, mean)
    cov = Statistics.cov(data)
    SemObs(data, cov, mean)
end

function SemObs(data, cov, mean::SemCalcMean)
    mean = vcat(Statistics.mean(data, dims = 1)...)
    SemObs(data, cov, mean)
end

function SemObs(data, cov::SemCalcCov, mean::SemCalcMean)
    cov = Statistics.cov(data)
    mean = vcat(Statistics.mean(data, dims = 1)...)
    SemObs(data, cov, mean)
end

function SemObs(; cov, mean = nothing)
    SemObs(nothing, cov, mean)
end

import Base.convert
convert(::Type{SemObs}, data) = SemObs(data)
convert(::Type{SemObs}, SemObs::SemObs) = SemObs

function model(ram, obs, par; objective = SemML(), optimizer = LBFGS())
    model(
    ram,
    convert(SemObs, obs),
    par,
    objective,
    optimizer,
    nothing,
    nothing,
    nothing,
    nothing)
end
