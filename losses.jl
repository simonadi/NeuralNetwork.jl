include("types.jl")

function bce_unit(y::Float32, label::Float32)
    label*log(y) + (1-label)*log(1-y)
end

function compute(loss::BinaryCrossEntropy, y::Array{Array{Float32, 1}, 1}, labels) # labels is a subarray, not sure how to type it cleanly
    y = [sample[1] for sample in y]
    labels = [sample[1] for sample in labels]
    -(1/length(labels))*sum(bce_unit.(y, labels))
end

function bce_d_unit(y::Float32, label::Float32)
    if y == label
        return 0
    else
        return (label-y)/(y*(y-1))
    end
end

function derivative(loss::BinaryCrossEntropy, y::Array{Float32}, labels::Array{Float32})
    bce_d_unit.(y,labels)
end
