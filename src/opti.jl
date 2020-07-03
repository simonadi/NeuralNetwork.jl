using LazyArrays
using Statistics

a = [rand(500) for _ in 1:10]
b = [rand(100) for _ in 1:10]

function f(a,b)
    mean(va*vb' for (va,vb) in zip(a,b))
end

function add!!(a, b)
    a .+ b
end

function sumop(f, g, x, y)
    z = zeros(Float32, size(x[1], 1), size(y[1], 1))
    for (x,y) in zip(x, y)
        z = @~ z + f(g(x,y))
    end
    z
end

function fop(a,b)
    l = length(a)
    sumop(x -> (@~ x ./ l), (x,y) -> (@~ x*y'), a, b)
end

function fo(a,b)
    z = zeros(Float32, size(x[1], 1), size(y[1], 1))
    for (x,y) in zip
end
