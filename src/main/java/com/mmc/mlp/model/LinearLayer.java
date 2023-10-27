package com.mmc.mlp.model;

import com.mmc.mlp.model.projection.NeuronProjection;
import com.mmc.mlp.model.projection.WeightProjection;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

class LinearLayer implements Layer {
    private final Neuron[] neurons;

    LinearLayer(int inSize, int outSize) {
        neurons = IntStream.range(0, outSize)
                .mapToObj(value -> new Neuron(inSize))
                .toArray(Neuron[]::new);
    }

    LinearLayer(List<NeuronProjection> neurons) {
        this.neurons = neurons.stream()
                .map(neuronProjection -> new Neuron(neuronProjection.getBias(), neuronProjection.getWeights()))
                .toArray(Neuron[]::new);
    }

    @Override
    public Tensor call(Tensor tensor) {
        Value[] out = new Value[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            out[i] = neurons[i].predict(tensor);
        }

        return new Tensor(out);
    }

    @Override
    public List<NeuronProjection> neuronProjections() {
        List<NeuronProjection> projections = new ArrayList<>();

        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];

            List<WeightProjection> weightProjections = new ArrayList<>();
            for (int w = 0; w < neuron.weights().size(); w++) {
                weightProjections.add(new WeightProjection(w, neuron.weights().weight(w).data()));
            }

            projections.add(new NeuronProjection(i, neuron.bias().data(), weightProjections));
        }

        return projections;
    }
}
