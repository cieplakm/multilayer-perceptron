package org.example.perc;

import java.util.stream.IntStream;

class LinearLayer implements Layer {
    Neuron[] neurons;

    LinearLayer(int inSize, int outSize) {
        neurons = IntStream.range(0, outSize)
                .mapToObj(value -> new Neuron(inSize))
                .toArray(Neuron[]::new);
    }

    public Tensor call(Tensor tensor) {
        Value[] out = new Value[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            out[i] = neurons[i].predict(tensor);
        }

        return new Tensor(out);
    }
}
