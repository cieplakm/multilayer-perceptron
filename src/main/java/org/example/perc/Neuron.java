package org.example.perc;

import java.util.Random;
import java.util.stream.DoubleStream;

class Neuron {

    Value learningRate = new Value(0.5);
    Value[] weights;
    Value bias;

    Neuron(int inSize) {
        Random random = new Random(3);
        this.weights = DoubleStream.generate(random::nextDouble)
                .limit(inSize)
                .mapToObj(Value::ofGradientable)
                .toArray(Value[]::new);
//        this.weights = new Tensor(-0.19, 0.24, -0.29, 0.23, 0.1).values;
        this.bias = Value.ofGradientable(random.nextDouble());
//        this.bias = Value.ofGradientable(-0.93);

        for (Value weight : weights) {
            weight.gradientApplicable = true;
        }
    }

    Value predict(Tensor tensor) {
        if (tensor.values.length != weights.length) {
            throw new IllegalArgumentException();
        }

        Value out = bias;

        Value[] ws = new Value[weights.length];

        for (int i = 0; i < weights.length; i++) {
            ws[i] = weights[i].multiply(tensor.values[i]);
            Value multiply = weights[i].multiply(tensor.values[i]);
            out = out.add(multiply);
        }

        Value tanh = out.tanh();

        return tanh;
    }
}
