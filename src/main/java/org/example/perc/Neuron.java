package org.example.perc;

import java.util.Random;
import java.util.stream.DoubleStream;

import static org.example.perc.MSE.loss;

class Neuron {

    Value learningRate = new Value(0.5);
    Value[] weights;
    Value bias;

    Neuron(int inSize) {
        Random random = new Random();
        this.weights = DoubleStream.generate(Math::random).limit(inSize).mapToObj(Value::new).toArray(Value[]::new);
        this.bias = new Value(Math.random());
    }

    Value predict(Tensor tensor) {
        if (tensor.values.length != weights.length) {
            throw new IllegalArgumentException();
        }

        Value out = bias;

        for (int i = 0; i < tensor.values.length; i++) {
            Value multiply = weights[i].multiply(tensor.values[i]);
            out = out.add(multiply);
        }

        return out.sigmoid();
    }

    public void train(Tensor tensor, Value target) {
        if (tensor.values.length != weights.length) {
            throw new IllegalArgumentException();
        }

        Value prediction = predict(tensor);
        Value error = target.subtract(prediction);
//        loss(prediction, target);
        Value biasBefore = bias;
        bias = bias.add(learningRate.multiply(error));

        for (int i = 0; i < weights.length; i++) {
            Value delta = learningRate.multiply(error).multiply(tensor.values[i]);
            Value sum = weights[i].add(delta);
            weights[i] = sum;
        }
    }
}
