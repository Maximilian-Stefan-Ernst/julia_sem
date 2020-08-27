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

*(F::Number, S::Symbol) = *(S,F)

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

#F*(I+A)*S*permutedims(I+A)*permutedims(F)

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
    0  0  0  0  0  0  0  0  0  0  0     :λ₁   0     0
    0  0  0  0  0  0  0  0  0  0  0     :λ₂  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     :λ₃  0
    0  0  0  0  0  0  0  0  0  0  0     0     :λ₄  0
    0  0  0  0  0  0  0  0  0  0  0     0     :λ₅  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :λ₆
    0  0  0  0  0  0  0  0  0  0  0     0     0     :λ₇
    0  0  0  0  0  0  0  0  0  0  0     0     0     :λ₈
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :lcov1  0     0
    0  0  0  0  0  0  0  0  0  0  0     :lcov2 :lcov3   0]


S =[:x1   0     0     0     0     0     0     0     0     0     0     0     0     0
      0     :x2   0     0     0     0     0     0     0     0     0     0     0     0
      0     0     :x3   0     0     0     0     0     0     0     0     0     0     0
      0     0     0     :x4   0     0     0     :x15  0     0     0     0     0     0
      0     0     0     0     :x5   0     :x16  0     :x17  0     0     0     0     0
      0     0     0     0     0     :x6   0     0     0     :x18  0     0     0     0
      0     0     0     0     :x16  0     :x7   0     0     0     :x19  0     0     0
      0     0     0     :x15 0      0     0     :x8   0     0     0     0     0     0
      0     0     0     0     :x17  0     0     0     :x9   0     :x20  0     0     0
      0     0     0     0     0     :x18  0     0     0     :x10  0     0     0     0
      0     0     0     0     0     0     :x19  0     :x20  0     :x11  0     0     0
      0     0     0     0     0     0     0     0     0     0     0     :x12  0     0
      0     0     0     0     0     0     0     0     0     0     0     0     :x13  0
      0     0     0     0     0     0     0     0     0     0     0     0     0     :x14]

    F =[1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]



@benchmark myzero(A^4)

SymMat = F*(I+A+A^2+A^3)*S*permutedims(I+A+A^2+A^3)*permutedims(F)

A^100

S =[Symbol(:(x[1])) 0 0 0
    0 Symbol(:(x[2])) 0 0
    0 0 Symbol(:(x[3])) 0
    0 0 0 Symbol(:(x[4]))]

F =[1 0 0 0
    0 1 0 0
    0 0 1 0]

A =[0 0 0 1
    0 0 0 Symbol(:(x[5]))
    0 0 0 Symbol(:(x[6]))
    0 0 0 0]

S*A

function myfun(x)
    return eval(Meta.parse("x[5]*x[2]"))
end

# user-supplied expression
expr = Meta.parse("x[5]*x[2]")

# splice into function body
@eval function myfun(x)
    x = x
    $expr
end

myfun(x)

myfun([1.0 1.0 1.0 1.0 4.0])

############## evaluate code

compute1 = [:(x[1]+x[2]) :(x[3]*x[2])
        :(x[1]+x[4]) :(x[1]*5.0*x[4])]

strf = Meta.parse("function f(x, A, compute)
    for i = (1:4)
        A[i] = compute[i]
    end
end")

eval(strf)


compute2 = :([x[1]+x[2] x[3]*x[2]
            x[1]+x[4] x[1]*5.0*x[4]])

expre = Meta.parse("x[1]+x[2]")

function wrap(par)
    A = Array{Float64}(undef, 2, 2)
    ft(par)
end

for i = 1:4
    A[i] = eval(compute1[i])
end

@generated function ft(x::Array{Float64, 1}, A)
    return :(A .= $compute1)
end

@generated function ft(x::Array{Float64, 1}, A)
    return :(for i = 1:length(A)
                A[i] = $compute1[i]
            end)
end

@generated function ft(x::Array{Float64, 2})
    return compute2
end

A = Array{Float64}(undef, 2, 2)

y = [1.0, 2, 3, 4, 3, 2, 5, 6]

ft(y, A)

ft(y)

using BenchmarkTools

@benchmark ft(y, A)

@benchmark ft(y)

@benchmark wrap(y)

@eval function imp_cov(x, compute)
    pre = Array{Expr}(undef, 2, 2)
    for i = (1:length(pre))
        pre[i] = $compute[i]
    end
    return pre
end

imp_cov(y, compute1)

j = imp_cov([1 1 2 3], compute)

test = compute[1]

@generated function imp_cov(x, cf)
    A = Array{Int64}(undef, 2, 2)
    return cf[1]
end

imp_cov([1 1 2 3], compute)

function imp_cov2(x, cov)
    pre = Array{eltype(x)}(undef, 2, 2)
    for i = (1:length(pre))
        pre[i] = @com_cov cov[i] x
    end
    return pre
end

imp_cov2([1 1 2 3], compute)

macro com_cov(term, x)
    return term
end

@com_cov compute[4] [1 1 2 3]
@com_cov compute[1]

map(eval, compute)

@benchmark A^30


# Go here to see my workaround for caculus package:
# https://github.com/johnmyleswhite/Calculus.jl/issues/111

using Calculus
using MacroTools: postwalk

Formula = Union{Expr,Symbol,Number}

function substitute(expr::T,pre,post) where {T<:Formula}
    d = Dict([i for i in zip(pre,post)])
    postwalk(x -> x isa Symbol && Base.isidentifier(x) && haskey(d,x) ? d[x] : x, expr)
end

substitute(expr::Array{T},pre,post) where {T<:Formula} = map(x->substitute(x,pre,post),expr)

import Base.convert
convert(::Type{Expr},arrexpr::T) where {T<:Formula} = arrexpr ### Would it work if it would be an integer?
convert(::Type{Expr},arrexpr::Array{T}) where {T<:Formula} = Expr(:call,:reshape,Expr(:vect,arrexpr...),:($(size(arrexpr))))

∇{T<:Formula}(expr::T,xs::Array{Symbol,1}) = Formula[differentiate(expr,xi) for xi in xs]
function ∇{T<:Formula}(expr::Array{T},xs::Array{Symbol,1})
    exprarr = Formula[]
    for expri in expr
        append!(exprarr,∇(expri,xs))
    end
    reshape(exprarr,(length(xs),size(expr)...))
end

### Now it is easy to define functions with @generated macro

expr = :((x^2 + y^2 + z^4 + y*z^2 + y*sin(x))*λ^8)
xs = [:x,:y,:z]

@generated f(x,λ) = substitute(convert(Expr,expr),xs,[:(x[$i]) for i in 1:length(xs)])
@generated ∇f(x,λ) = substitute(convert(Expr,∇(expr,xs)),xs,[:(x[$i]) for i in 1:length(xs)])
@generated ∇∇f(x,λ) = substitute(convert(Expr,∇(∇(expr,xs),xs)),xs,[:(x[$i]) for i in 1:length(xs)])
@generated ∇∇∇f(x,λ) = substitute(convert(Expr,∇(∇(∇(expr,xs),xs),xs)),xs,[:(x[$i]) for i in 1:length(xs)])

y = [1.0, 2, 3, 4, 3, 2, 5, 6]

f(y, 1.0)


### Symengine

using LinearAlgebra
using SymEngine

@vars x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11
for i = 1:11 symbols("x_$i") end

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x21   0     0
    0  0  0  0  0  0  0  0  0  0  0     x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     x5  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x6
    0  0  0  0  0  0  0  0  0  0  0     0     0     x7
    0  0  0  0  0  0  0  0  0  0  0     0     0     x8
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x9  0     0
    0  0  0  0  0  0  0  0  0  0  0     x10 x11   0]

S =[x1   0     0     0     0     0     0     0     0     0     0     0     0     0
      0     x2   0     0     0     0     0     0     0     0     0     0     0     0
      0     0     x3   0     0     0     0     0     0     0     0     0     0     0
      0     0     0     x4   0     0     0     x15  0     0     0     0     0     0
      0     0     0     0     x5   0     x16  0     x17  0     0     0     0     0
      0     0     0     0     0     x6   0     0     0     x18  0     0     0     0
      0     0     0     0     x16  0     x7   0     0     0     x19  0     0     0
      0     0     0     x15 0      0     0     x8   0     0     0     0     0     0
      0     0     0     0     x17  0     0     0     x9   0     x20  0     0     0
      0     0     0     0     0     x18  0     0     0     x10  0     0     0     0
      0     0     0     0     0     0     x19  0     x20  0     x11  0     0     0
      0     0     0     0     0     0     0     0     0     0     0     x12  0     0
      0     0     0     0     0     0     0     0     0     0     0     0     x13  0
      0     0     0     0     0     0     0     0     0     0     0     0     0     x14]

    F =[1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]

inv(A)

Matrix(inv(SymEngine.CDenseMatrix(1.0I-A)))




compute1 = [x1+x2 x3*x2
            x1+x4 x1*5.0*x4]

SymEngine.toString.(compute1)

parnames = Dict(
    "x1" => "x[1]",
    "x2" => "x[2]",
    "x3" => "x[3]",
    "x4" => "x[4]")

implied = SymEngine.toString.(compute1)

for i in 1:length(implied)
    for j in 1:length(parnames)
        implied[i] = reduce(replace, parnames, init = implied[i])
    end
end

func = ""
function merge_fun(func, implied)
    for i = 1:length(implied)
        func *= "pre[$i] = "*"$(implied[i]) ;"
    end
    return func
end

func = merge_fun(func, implied)

eval(Meta.parse("function f(x, pre)"*func*"end"))

myfunc = f

function parse_func(implied, parnames)
    implies = SymEngine.toString.(compute1)
end

A = rand(2,2)

function test(x, pre)
    f(x, pre)
end

@benchmark test([1.0 2 3 4], A)


### MWE

x = [1.0 2 3 4]

compute = ["x[1]+x[2]" "x[3]*x[2]"
            "x[1]+x[4]" "x[1]*5.0*x[4]"]

#
func_str = ""

function merge_fun(func_str, implied)
    for i = 1:length(implied)
        func_str *= "pre[$i] = "*"$(implied[i]) ;"
    end
    return func_str
end

func_str = merge_fun(func, compute)

eval(Meta.parse("function myfun(x, pre)"*func*"end"))

# benchmark

A = rand(2,2)

@benchmark myfun(x, A)


compute_expr = Meta.parse.(compute)

compute_expr = Expr(Symbol(compute_expr))

compute

compute2 = "["

function merge_fun2(compute, compute2)
    for i in 1:size(compute, 1)
        for j in 1:size(compute, 2)
            compute2 *= " "*compute[i,j]
        end
        compute2 *=";"
    end
    return compute2
end

compute2 = merge_fun2(compute, compute2)

compute2 = chop(compute2)*"]"

compute_expr = Meta.parse(compute2)

@generated function myfun2(x)
    return compute_expr
end

@benchmark myfun2(x)


### ODE Solution
using ModelingToolkit, BenchmarkTools

@variables t σ[1:4,1:4](t)
@derivatives D'~t
eqs = Array{Equation}(undef,4,4)
for i in 1:4
    for j in 1:4
        eqs[i,j] =  D(σ[i,j]) ~ 1 + σ[i,j]
    end
end

sys = ODESystem(vec(eqs))
f = ODEFunction(sys)

u = rand(4,4)
du = similar(u)
@benchmark f(du,u,nothing,0.0)


### simple test
@variables x y z
@variables x1 x2 x3 x4

compute = [x1 + x2 x3*x2
            x1 + x4 x1*5.0*x4]

f1, f2 = eval.(ModelingToolkit.build_function(x*y, [x,y]))

f1, f2 = eval.(ModelingToolkit.build_function(compute, [x1 x2 x3 x4]))

A = Array{Float64}(undef, 2, 2)

simplify_constants.(inv(compute))

@benchmark f1([1.0 2 3 4])

inv(compute)

compute = [x1 + x2 x3*x2
            x1 + x4 x1*5.0*x4]

ModelingToolkit.build_function(compute, [x1 x2 x3 x4])

@benchmark f2(A, [1.0 2 3 4])

compute2 = :([x[1]+x[2] x[3]*x[2]
            x[1]+x[4] x[1]*5.0*x[4]])

@generated function f3(x)
    return compute2
end


@benchmark f3([1.0 2 3 4])


### test for big model
using ModelingToolkit, BenchmarkTools, LinearAlgebra, SymbolicUtils

@variables x[1:31]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21]   0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22]  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25]  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29]  0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31]   0]



const B = rand(100, 100)

@benchmark det(B)

S =[x[1]   0     0     0     0     0     0     0     0     0     0     0     0     0
      0     x[2]   0     0     0     0     0     0     0     0     0     0     0     0
      0     0     x[3]   0     0     0     0     0     0     0     0     0     0     0
      0     0     0     x[4]   0     0     0     x[15]  0     0     0     0     0     0
      0     0     0     0     x[5]   0     x[16]  0     x[17]  0     0     0     0     0
      0     0     0     0     0     x[6]   0     0     0     x[18]  0     0     0     0
      0     0     0     0     x[16]  0     x[7]   0     0     0     x[19]  0     0     0
      0     0     0     x[15] 0      0     0     x[8]   0     0     0     0     0     0
      0     0     0     0     x[17]  0     0     0     x[9]   0     x[20]  0     0     0
      0     0     0     0     0     x[18]  0     0     0     x[10]  0     0     0     0
      0     0     0     0     0     0     x[19]  0     x[20]  0     x[11]  0     0     0
      0     0     0     0     0     0     0     0     0     0     0     x[12]  0     0
      0     0     0     0     0     0     0     0     0     0     0     0     x[13]  0
      0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

    F =[1 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0 0]


covmat = simplify_constants.(F*simplify_constants.(1.0I+A+A^2+A^3)*S*
            permutedims(simplify_constants.(1.0I+A+A^2+A^3))*permutedims(F))

test = simplify_constants.(A^4)


f1, f2 = eval.(ModelingToolkit.build_function(covmat, x))

ModelingToolkit.build_function(covmat, x)

start = vcat(fill(1.0, 11), fill(0.05, 3), fill(0.0, 6), fill(1.0, 8), fill(0, 3))

start = rand(31)

preal = zeros(size(covmat))

f2(preal, start)

preal

@benchmark f2(preal, start)

function imp_cov(matrices)
      invia = inv(I - matrices[3]) # invers of I(dentity) minus A matrix
      imp = matrices[2]*invia*matrices[1]*transpose(invia)*transpose(matrices[2])
      return imp
end

x = start


A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[21]   0     0
    0  0  0  0  0  0  0  0  0  0  0     x[22]  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[23]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[24]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[25]  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[26]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[27]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[28]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[29]  0     0
    0  0  0  0  0  0  0  0  0  0  0     x[30] x[31]   0]

S =[x[1]   0     0     0     0     0     0     0     0     0     0     0     0     0
      0     x[2]   0     0     0     0     0     0     0     0     0     0     0     0
      0     0     x[3]   0     0     0     0     0     0     0     0     0     0     0
      0     0     0     x[4]   0     0     0     x[15]  0     0     0     0     0     0
      0     0     0     0     x[5]   0     x[16]  0     x[17]  0     0     0     0     0
      0     0     0     0     0     x[6]   0     0     0     x[18]  0     0     0     0
      0     0     0     0     x[16]  0     x[7]   0     0     0     x[19]  0     0     0
      0     0     0     x[15] 0      0     0     x[8]   0     0     0     0     0     0
      0     0     0     0     x[17]  0     0     0     x[9]   0     x[20]  0     0     0
      0     0     0     0     0     x[18]  0     0     0     x[10]  0     0     0     0
      0     0     0     0     0     0     x[19]  0     x[20]  0     x[11]  0     0     0
      0     0     0     0     0     0     0     0     0     0     0     x[12]  0     0
      0     0     0     0     0     0     0     0     0     0     0     0     x[13]  0
      0     0     0     0     0     0     0     0     0     0     0     0     0     x[14]]

test = imp_cov((S, F, A))

@benchmark test = imp_cov((S, F, A))

preal = zeros(size(covmat))

@benchmark f2(preal, start)

preal

### MWEs

using ModelingToolkit, SymbolicUtils, LinearAlgebra

@variables x[1:11]

A =[0  0  0  0  0  0  0  0  0  0  0     1     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[1]  0     0
    0  0  0  0  0  0  0  0  0  0  0     x[2]  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1     0
    0  0  0  0  0  0  0  0  0  0  0     0     x[3]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[4]  0
    0  0  0  0  0  0  0  0  0  0  0     0     x[5]  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[6]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[7]
    0  0  0  0  0  0  0  0  0  0  0     0     0     x[8]
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x[9]  0     0
    0  0  0  0  0  0  0  0  0  0  0     x[10] x[11] 0]


inv(1.0I-A)

A^3

@syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11

A =[0  0  0  0  0  0  0  0  0  0  0     1    0     0
    0  0  0  0  0  0  0  0  0  0  0     x1   0      0
    0  0  0  0  0  0  0  0  0  0  0     x2   0      0
    0  0  0  0  0  0  0  0  0  0  0     0     1    0
    0  0  0  0  0  0  0  0  0  0  0     0     x3   0
    0  0  0  0  0  0  0  0  0  0  0     0     x4   0
    0  0  0  0  0  0  0  0  0  0  0     0     x5   0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     x6
    0  0  0  0  0  0  0  0  0  0  0     0     0     x7
    0  0  0  0  0  0  0  0  0  0  0     0     0     x8
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     x9  0     0
    0  0  0  0  0  0  0  0  0  0  0     x10 x11 0]

A^4

inv(1.0I-A)

A = [x1 x2
    x3 x4]

inv(A)

function f(x)
    for i = 1:10^8
        x = i
    end
    return(x)
end

f(x)

@benchmark f(x)


A = [0.5 0.5 0
    0.5 -0.5 0
    0 1 1]

inv(A)

a = 0

function f(a)
    for i in 1:552
        a = a + 552/(553 - i)
    end
    return a
end

f(a)
