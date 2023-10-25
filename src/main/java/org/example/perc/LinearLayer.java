package org.example.perc;

import java.util.stream.IntStream;

class LinearLayer implements Layer {
    Neuron[] neurons;

    LinearLayer(int inSize, int outSize) {
        neurons = IntStream.range(0, outSize)
                .mapToObj(value -> new Neuron(inSize))
                .toArray(Neuron[]::new);
        System.out.println("-------neuron init------");
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            String s = "Neuron["+i+"] B=" + neuron.bias.data;
            for (int j = 0; j < neuron.weights.length; j++) {
                s = s + " W[" + j + "]=" + neuron.weights[j].data;
            }
            System.out.println(s);
        }
    }

    public Tensor call(Tensor tensor) {
        Value[] out = new Value[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            out[i] = neurons[i].predict(tensor);
        }

        return new Tensor(out);
    }

    public void trainNeuron(Tensor tensor, Value target) {
        for (Neuron neuron : neurons) {
            neuron.train(tensor, target);
        }

        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            String s = "Neuron["+i+"] B=" + neuron.bias.data;
            for (int j = 0; j < neuron.weights.length; j++) {
                s = s + " W[" + j + "]=" + neuron.weights[j].data;
            }
            System.out.println(s);
        }
    }
}
