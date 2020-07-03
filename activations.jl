include("types.jl")


# Sigmoid activation

sigmoid_f(x::Float32) = 1/(1+exp(-x))

# Apply the sigmoid on all neurons of the layer
compute(activation::Sigmoid, x::Array{Float32}) = sigmoid_f.(x)

sigmoid_d_f(x::Number) = exp(-x)/((1+exp(-x))^2)

# Apply the sigmoid derivative on all neurons of the layer
derivative(activation::Sigmoid, x::Array{Float32}) = sigmoid_d_f.(x)


# ReLu activation

compute(activation::ReLu, x::Array{Number}) = max.(0, x)

derivative(activation::ReLu, x::Array{Number}) = x .> 0

# Unit activation

compute(activation::Unit, x::Array{T}) where T<:Number = x

derivative(activation::Unit, x::Array{T}) where T<:Number = 1
