include("types.jl")

function accuracy_unit(y, label)
    Int8(round.(y)==label)
end

function accuracy(y, labels)
    sum(accuracy_unit.(y, labels))/length(labels)
end
