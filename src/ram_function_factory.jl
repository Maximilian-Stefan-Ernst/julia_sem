S =[:x1 0 0 0
    0 :x2 0 0
    0 0 :x3 0
    0 0 0 :x4]

F =[1 0 0 0
    0 1 0 0
    0 0 1 0]

A =[0 0 0 1
    0 0 0 :λ₂
    0 0 0 :λ₃
    0 0 0 0]


using LinearAlgebra
import Base: +, *, zero, one, convert

function *(S::Symbol, F::Number)
    if (iszero(F))| (S == Symbol(0))
        return Symbol(0)
    elseif isone(F)
        return S
    elseif S == Symbol(1)
        return Symbol(F)
    else
            return Symbol(S, *, F)
    end
end

function *(F::Number, S::Symbol)
    if (iszero(F))| (S == Symbol(0))
        return Symbol(0)
    elseif isone(F)
        return S
    elseif S == Symbol(1)
        return Symbol(F)
    else
            return Symbol(S, *, F)
    end
end

function *(S::Symbol, F::Symbol)
    if (F == Symbol(0))|(S == Symbol(0))
        return Symbol(0)
    elseif F == Symbol(1)
        return S
    elseif S == Symbol(1)
        return Symbol(F)
    else
        return Symbol(S, *, F)
    end
end

function +(F::Number, S::Symbol)
    if iszero(F)
        return S
    elseif S == Symbol(0)
        return Symbol(F)
    else
        return Symbol(F, +, S)
    end
end

function +(S::Symbol, F::Number)
    if iszero(F)
        return S
    elseif S == Symbol(0)
        return Symbol(F)
    else
        return Symbol(F, +, S)
    end
end

function +(F::Symbol, S::Symbol)
    if F == Symbol(0)
        return S
    elseif S == Symbol(0)
        return Symbol(F)
    else
        return Symbol(F, +, S)
    end
end

-(F::Number, S::Symbol) = Symbol(F, -, S)
-(S::Symbol, F::Number) = Symbol(S, -, F)
-(F::Symbol, S::Symbol) = Symbol(F, -, S)

one(::Type{Any}) = Symbol(1)
zero(a::Symbol) = Symbol(0)

F*(I+A)*S*permutedims(I+A)*permutedims(F)

convert(::Type{Symbol}, x::Number) = Symbol(x)

function myequal(A, B)
    return convert(Array{Symbol}, A) == convert(Array{Symbol}, B)
end

myequal(I+A, I+A+A^2)

function myzero(A)
    return convert(Array{Symbol}, A) == convert(Array{Symbol}, zeros(Int64, size(A)))
end

myzero(A^2)

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30 :x31   0]


@benchmark myzero(A^4)
