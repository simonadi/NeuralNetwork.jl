import Base.widen

abstract type ActivationFunction end
struct Sigmoid <: ActivationFunction end
Base.Broadcast.broadcastable(ac::Sigmoid) = Ref(ac)

struct Unit <: ActivationFunction end
Base.Broadcast.broadcastable(ac::Unit) = Ref(ac)

struct ReLu <: ActivationFunction end
Base.Broadcast.broadcastable(ac::ReLu) = Ref(ac)

abstract type Loss end
struct BinaryCrossEntropy <: Loss end
Base.Broadcast.broadcastable(loss::BinaryCrossEntropy) = Ref(loss)

abstract type Metrics end
struct Accuracy <: Metrics end

abstract type Layer end
