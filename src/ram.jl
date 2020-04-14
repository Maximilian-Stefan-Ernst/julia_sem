using BenchmarkTools, ForwardDiff, Random

# test preallocation of typed array for rams
struct Ram{S <: AbstractArray,
    F <: AbstractArray,
    A <: AbstractArray,
    M <: AbstractVecOrMat}
    s::S
    f::F
    a::A
    m::M
end

n = 10
s = rand(n, n)
f = rand(n, n)
a = rand(n, n)
m = rand(n)

test = Ram(s, f, a, m)

function(ram::Ram)(parameters)
    T = eltype(parameters)
    s = similar(ram.s, T)
    f = similar(ram.f, T)
    a = similar(ram.a, T)
    m = similar(ram.m, T)
    return Ram(s, f, a, m)
end

test(collect(1:10))
@benchmark test(collect(1:10))
test(ForwardDiff.Dual(10))
@benchmark test(ForwardDiff.Dual(10))
@code_warntype test(ForwardDiff.Dual(10))

n = 10
s = zeros(n, n)
f = zeros(n, n)
a = zeros(n, n)
m = zeros(n)

test = Ram(s, f, a, m)

A = test(ForwardDiff.Dual(10))

convert(Type{Array{typeof(ForwardDiff.Dual(10))}}, [5.0 5.0])

# test inplace assignment with BitArray

bit = BitArray([0 0 0
         1 0 0
         1 1 0])
mat = reshape(collect(1:9), 3, 3)
function repl(mat, bit, replace)
    mat[bit] .= replace
end
repl(mat, bit, 4)
mat
@benchmark repl(mat, bit, 4)

repl(mat, bit, collect(1:3))
mat
@benchmark repl(mat, bit, collect(1:3))


# test breaking of vector into appropriate size
par = collect(1:50)

@benchmark t = par[1:sum(bit)]


# bring it all together
using BenchmarkTools, ForwardDiff, Random

struct Ram{S <: AbstractArray,
    F <: AbstractArray,
    A <: AbstractArray,
    M <: AbstractVecOrMat,
    SF <: AbstractArray,
    FF <: AbstractArray,
    AF <: AbstractArray,
    MF <: AbstractVecOrMat} # F for Free
    s::S
    f::F
    a::A
    m::M
    sfree::SF
    ffree::FF
    afree::AF
    mfree::MF
end

n = 10
s = rand(n, n)
f = rand(n, n)
a = rand(n, n)
m = rand(n)
sfree = bitrand(n, n)
ffree = bitrand(n, n)
afree = bitrand(n, n)
mfree = bitrand(n)
sumfree = sum(sfree) + sum(ffree) + sum(afree) + sum(mfree)

test = Ram(s, f, a, m, sfree, ffree , afree, mfree)

function(ram::Ram)(parameters)
    T = eltype(parameters)
    s = similar(ram.s, T)
    f = similar(ram.f, T)
    a = similar(ram.a, T)
    m = similar(ram.m, T)

    snfree = sum(ram.sfree)
    start = one(snfree)
    if snfree > zero(snfree)
        s[ram.sfree] .= parameters[start:(start + snfree - one(snfree))]
        start = start + snfree
    end

    fnfree = sum(ram.ffree)
    if fnfree > zero(fnfree)
        f[ram.ffree] .= parameters[start:(start + fnfree - one(fnfree))]
        start = start + fnfree
    end

    anfree = sum(ram.afree)
    if anfree > zero(anfree)
        a[ram.afree] .= parameters[start:(start + anfree - one(anfree))]
        start = start + anfree
    end

    mnfree = sum(ram.mfree)
    if fnfree > zero(fnfree)
        m[ram.mfree] .= parameters[start:(start + mnfree - one(mnfree))]
    end
    return Ram(s, f, a, m, ram.sfree, ram.ffree, ram.afree, ram.mfree)
end

@benchmark test(rand(Int, sumfree)//3)
# 1.23 ms for n = 100; 30100 elements
# 18 μs for n = 10; 310
