function split_batches(array::Array, batch_size::UInt16)
    Iterators.partition(array, batch_size)
end

function sumop(f, g, x, y)
    z = zeros(Float32, size(x[1], 1), size(y[1], 1))
    for (a,b) in zip(x, y)
        z = @~ z + f(g(a,b))
    end
    z
end

function fop(a,b)
    l = length(a)
    sumop(x -> (@~ x ./ l), (x,y) -> (@~ x*y'), a, b)
end
