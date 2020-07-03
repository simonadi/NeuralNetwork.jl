import Statistics: mean
import LazyArrays: @~, materialize

include("../types.jl")
include("../utils.jl")

mutable struct Dense{T<:ActivationFunction} <: Layer
    input_size::UInt32
    output_size::UInt32
    activation::T
    weights::Array{Array{Float32, 2}}
    bias::Array{Array{Float32, 1}}

    Dense(isize, osize) = new{typeof(Unit())}(isize, osize, Unit(), [rand(Float32, osize, isize)*0.01], [zeros(Float32, osize)])
    Dense(isize, osize, acfunc::T) where {T} = new{T}(isize, osize, acfunc, [rand(Float32, osize, isize)*0.01], [zeros(Float32, osize)])
end

Base.Broadcast.broadcastable(l::Dense) = Ref(l)

function forward(layer::Dense, input) where T<:Real # TODO : add type of input
    layer.weights .* input .+ layer.bias
end

function input_error(layer::Dense, output_error::Array)
    [transpose(layer.weights[1])] .* output_error
end

function weights_error(layer::Dense, output_error::Array, input::Array)
    mean(output*transpose(in) for (output, in) in zip(output_error, input))
    # input = transpose.(input)
    # output_error .* input
    # fop(output_error, input)
end

function layer_error(layer::Dense, output_noac::Array, error::Array)
    error .* derivative(layer.activation, output_noac)
end

function bias_error(layer::Dense, output_error::Array)
    mean(output_error)
end

function update(layer::Dense, grad_weights, grad_bias::Array, rate)
    layer.weights[1] -= rate*grad_weights
    # layer.weights[1] -= rate*materialize(grad_weights)
    layer.bias[1] -= rate*grad_bias
end
