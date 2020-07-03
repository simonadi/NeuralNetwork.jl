using .NeuralNetwork
using CSV
using Profile

df = CSV.read("examples/test_set/input.csv", header=false)
input = convert(Matrix{Float32}, df)

input = [input[i,:] for i in 1:2000]

df = CSV.read("examples/test_set/labels.csv", header=false)
labels = convert(Matrix{Float32}, df)

labels = [[labels[i,1]] for i in 1:2000]

bce = BinaryCrossEntropy()
sigmoid = Sigmoid()

# n = Network(layers, bce)

# train(n, input, labels, batch_size=100)

layers = [Dense(1001, 500), Dense(500, 250), Dense(250, 100), Dense(100, 50), Dense(50,1, sigmoid)]

function run()
    n = Network(layers, bce)
    train(n, input, labels, batch_size=100, epochs=10)
end

run()

@profile run()

f = open("profile.txt", "w")
Profile.print(IOContext(f, :displaysize => (24,500)), sortedby=:count)
close(f)
