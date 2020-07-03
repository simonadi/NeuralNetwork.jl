include("layers/dense.jl")
include("losses.jl")
include("metrics.jl")
include("activations.jl")

using Dates

struct Network
    layers::Array{Layer}
    loss::Loss
    metrics::Array{Metrics}
    Network(layers, loss) = new(layers, loss, [])
end


function train(network::Network, input::Array{Array{Float32,1},1}, labels::Array{Array{Float32,1},1}; epochs::Number = 10, batch_size::Number = 32)
    input, labels = Iterators.partition(input, batch_size), Iterators.partition(labels, batch_size)
    tstart = nothing
    for epoch in 1:epochs
        println("Epoch $(epoch)/$(epochs)")
        if epoch == 2
            tstart = now()
        end
        forward_data = nothing
        last_batch_label = nothing
        for (batch_input, batch_label) in zip(input, labels)
            output_noac = [copy(batch_input)]
            output_ac = Array[]
            forward_data = batch_input
            for l in network.layers
                output = forward(l, forward_data)
                forward_data = compute.(l.activation, output)
                push!(output_noac, output)
                push!(output_ac, forward_data)
            end
            loss = derivative.(network.loss, forward_data, batch_label)
            gradients_weights = []
            # gradients_bias::Array{Number} = []
            gradients_bias = []
            out_noac = pop!(output_noac)
            for l in reverse(network.layers)
                error = layer_error.(l, out_noac, loss)
                out_noac = pop!(output_noac)
                push!(gradients_weights, weights_error(l, error, out_noac))
                push!(gradients_bias, bias_error(l, error))
                loss = input_error(l, error)
            end

            for (l, grad_w, grad_b) in zip(network.layers, reverse(gradients_weights), reverse(gradients_bias))
                update(l, grad_w, grad_b, 0.1)
            end
            last_batch_label = batch_label
        end
        loss = mean(compute(network.loss, forward_data, last_batch_label))
        acc = mean(accuracy(forward_data, last_batch_label))
        println("Loss : $(loss)")
        println("Accuracy : $(acc)")
    end
    elapsed = now() - tstart
    println("Time from second epoch : $(elapsed)")
end
