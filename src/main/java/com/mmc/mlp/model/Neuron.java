package com.mmc.mlp.model;

import com.mmc.mlp.model.projection.WeightProjection;

import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

class Neuron {
    private final Weights weights;
    private Value bias;

    Neuron(int inSize) {
        Random random = new Random(3);

        Value[] values = DoubleStream.generate(random::nextDouble)
                .limit(inSize)
                .mapToObj(Value::ofGradientable)
                .toArray(Value[]::new);

        this.weights =  new Weights(values);
        this.bias = Value.ofGradientable(random.nextDouble());
    }

    Neuron(double bias, List<WeightProjection> weights) {
        Value[] values = weights.stream()
                .map(weightProjection -> new Value(weightProjection.getWeight()))
                .toArray(Value[]::new);

        this.weights =  new Weights(values);
        this.bias = new Value(bias);
    }

    Value bias() {
        return bias;
    }

    Weights weights() {
        return weights;
    }

    Value predict(Tensor tensor) {
        if (tensor.size() != weights.size()) {
            throw new IllegalArgumentException();
        }

        Value out = bias;

        for (int i = 0; i < weights.size(); i++) {
            Value multiply = weights.weight(i).multiply(tensor.valueAt(i));
            out = out.add(multiply);
        }

        return out.tanh();
    }
}
