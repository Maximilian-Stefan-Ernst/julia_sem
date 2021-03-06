struct ∇SemML{
        X <: AbstractArray,
        Y <: AbstractArray,
        Z <: AbstractArray,
        W,
        F,
        F2,
        I,
        I2,
        T,
        F3,
        TM,
        F4,
        TBm,
        I3} <: DiffFunction
    B::X
    B!::F
    E::Y
    E!::F2
    F::Z
    C::W
    S_ind_vec::I
    A_ind_vec::I2
    matsize::T
    M!::F3
    M::TM
    Bm!::F4
    Bm::TBm
    M_ind_vec::I3
end

function ∇SemML(
        A::Spa1,
        S::Spa2,
        F::Spa3,
        parameters,
        start_val;
        M = nothing
            ) where {
            Spa1 <: SparseMatrixCSC,
            Spa2 <: SparseMatrixCSC,
            Spa3 <: SparseMatrixCSC
            }

    A = copy(A)
    S = copy(S)
    F = copy(F)


    invia = neumann_series(A)

    #imp_cov_sym = F*invia*S*permutedims(invia)*permutedims(F)
    #imp_cov_sym = Array(imp_cov_sym)
    invia = ModelingToolkit.simplify.(invia)
    B = invia
    E = B*S*B'
    E = E*permutedims(F)
    E = Array(E)
    E = ModelingToolkit.simplify.(E)
    B = F*B
    B = Array(B)
    B = ModelingToolkit.simplify.(B)

    B! =
        eval(ModelingToolkit.build_function(
            B,
            parameters
            )[2])

    E! =
        eval(ModelingToolkit.build_function(
            E,
            parameters
            )[2])

    B_pre = zeros(size(F)...)
    E_pre = zeros(size(F, 2), size(F, 1))

    grad = similar(start_val)
    matsize = size(A)
    C_pre = zeros(size(F, 1), size(F, 1))

    #S_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    #A_ind_vec = Vector{Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}()
    S_ind_vec = Vector{Vector{CartesianIndex{2}}}()
    A_ind_vec = Vector{Vector{CartesianIndex{2}}}()

    for i = 1:size(parameters, 1)
        S_ind = findall(var -> isequal(var, parameters[i]), S)
        A_ind = findall(var -> isequal(var, parameters[i]), A)

        push!(S_ind_vec, S_ind)
        push!(A_ind_vec, A_ind)
    end


    if !isnothing(M)
        
        fun_mean =
            eval(ModelingToolkit.build_function(
                M,
                parameters
            )[2])
        M_pre = zeros(size(F)[2])

        Bm = B*M
        Bm = ModelingToolkit.simplify.(Bm)
        fun_bm =
            eval(ModelingToolkit.build_function(
                Bm,
                parameters
            )[2])
        Bm_pre = zeros(size(F)[2])

        M_ind_vec = Vector{Vector{Int64}}()
        for i = 1:size(parameters, 1)
            M_ind = findall(var -> isequal(var, parameters[i]), M)
            push!(M_ind_vec, M_ind)
        end
    else
        M_ind_vec = nothing
        M_pre = nothing
        Bm_pre = nothing
        fun_mean = nothing
        fun_bm = nothing
    end

    return ∇SemML(
        B_pre,
        B!,
        E_pre,
        E!,
        F,
        C_pre,
        S_ind_vec,
        A_ind_vec,
        matsize,
        fun_mean,
        M_pre,
        fun_bm,
        Bm_pre,
        M_ind_vec)
end

function (diff::∇SemML)(par, grad, model::Sem{O, I, L, D}) where
    {O <: SemObs, L <: Loss, I <: Imply, D <: SemAnalyticDiff}
    a = cholesky(Hermitian(model.imply.imp_cov); check = false)
    if !isposdef(a)
        grad .= 0
    else
        #ld = logdet(a)
        Σ_inv = inv(a)
        diff.B!(diff.B, par) # B = inv(I-A)
        diff.E!(diff.E, par) # E = B*S*B'
        if !isnothing(diff.M)
            b = model.observed.obs_mean - model.imply.imp_mean
        end
        let B = diff.B, E = diff.E, F = diff.F, D = model.observed.obs_cov,
            Bm = diff.Bm
            #C = LinearAlgebra.I-Σ_inv*D
            mul!(diff.C, Σ_inv, D)
            diff.C .= LinearAlgebra.I-diff.C
            for i = 1:size(par, 1)
                term = similar(diff.C)
                term2 = similar(diff.C)
                sparse_outer_mul!(term, B, E, diff.A_ind_vec[i])
                sparse_outer_mul!(term2, B, B', diff.S_ind_vec[i])
                Σ_der = term2 + term + term'
                grad[i] = tr(Σ_inv*Σ_der*diff.C)
                # if !isnothing(diff.M)
                #     term3 = Vector{Float64}(undef, size(B, 1))
                #     term4 = similar(term3)
                #     sparse_outer_mul!(term3, B, Bm, diff.A_ind_vec[i])
                #     sparse_outer_mul!(term4, B, diff.M_ind_vec[i])
                #     µ_der = term3 + term4
                #     gradupdate = (b'*Σ_inv*Σ_der + 2*µ_der')*Σ_inv*b
                #     grad[i] -= gradupdate[1]
                # end
            end
        end
    end
end