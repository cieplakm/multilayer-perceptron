package org.example.perc.model;

import org.example.perc.projection.WeightProjection;

import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

class Neuron {
    Value[] weights;
    Value bias;

    Neuron(int inSize) {
        Random random = new Random(3);
        this.weights = DoubleStream.generate(random::nextDouble)
                .limit(inSize)
                .mapToObj(Value::ofGradientable)
                .toArray(Value[]::new);
        this.bias = Value.ofGradientable(random.nextDouble());
    }

    public Neuron(double bias, List<WeightProjection> weights) {
        this.bias = new Value(bias);
        this.weights = weights.stream()
                .map(weightProjection -> new Value(weightProjection.getWeight()))
                .toArray(Value[]::new);
    }

    Value predict(Tensor tensor) {
        if (tensor.size() != weights.length) {
            throw new IllegalArgumentException();
        }

        Value out = bias;

        Value[] ws = new Value[weights.length];

        for (int i = 0; i < weights.length; i++) {
            ws[i] = weights[i].multiply(tensor.values(i));
            Value multiply = weights[i].multiply(tensor.values(i));
            out = out.add(multiply);
        }

        Value tanh = out.tanh();

        return tanh;
    }
}
