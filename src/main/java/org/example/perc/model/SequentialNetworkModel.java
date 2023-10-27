package org.example.perc.model;

import org.example.perc.projection.LayerProjection;
import org.example.perc.projection.NetworkProjection;

import java.util.ArrayList;
import java.util.List;

import static org.example.perc.model.MSE.loss;

public class SequentialNetworkModel {
    private String name;
    private Value learningRate;
    private final Layer[] layers;


    SequentialNetworkModel(String name, Value learningRate, Layer... layers) {
        this.name = name;
        this.learningRate = learningRate;
        this.layers = layers;
    }

    SequentialNetworkModel(String name, int numOfLayers, int inputSize, int outputSize, int[] layerOutputs, double learningRate) {
        if (numOfLayers > 1 && layerOutputs.length != numOfLayers - 1) {
            throw new RuntimeException("Wrong seq definition");
        }
        this.name = name;
        this.learningRate = Value.of(learningRate);
        this.layers = new Layer[numOfLayers];

        for (int layer = 0; layer < numOfLayers; layer++) {
            this.layers[layer] = new LinearLayer(inLayerSize(layer, inputSize, layerOutputs), outLayerSize(numOfLayers, layer, outputSize, layerOutputs));
        }
    }

    private int inLayerSize(int layer, int inputSize, int[] doubles) {
        if (layer == 0) {
            return inputSize;
        }

        return doubles[layer - 1];
    }

    private static int outLayerSize(int numOfLayers, int layer, int outputSize, int[] doubles) {
        if (layer == numOfLayers - 1) {
            return outputSize;
        }

        if (layer == 0) {
            return doubles[layer];
        }

        return doubles[layer];
    }

    public Tensor predict(Tensor data) {
        for (Layer layer : layers) {
            data = layer.call(data);
        }

        return data;
    }

    public void train(int epoch, TreningSet[] treningSets, boolean logging, Double stopWhenError) {
        long t1 = System.currentTimeMillis();
        for (int i = 0; i < epoch; i++) {
            for (TreningSet data : treningSets) {
                Tensor prediction = predict(data.getData());
                for (int k = 0; k < prediction.size(); k++) {
                    Value loss = loss(prediction.values(k), data.getTarget().values(k), logging);
                    loss.gradientOne();
                    backPropagate(loss);
                    applyGradient(loss);
                    zeroGradient(loss);
                }
            }
        }

        long t2 = System.currentTimeMillis();
        System.out.println(String.format("Trening %s epoch done in %sms", epoch, t2 - t1));

    }

    private void backPropagate(Value value) {
        value.backward();

        if (value.parentA() != null) {
            backPropagate(value.parentA());
        }
        if (value.parentB() != null) {
            backPropagate(value.parentB());
        }
    }

    private void applyGradient(Value value) {
        value.applyGrad(learningRate);

        if (value.parentA() != null) {
            applyGradient(value.parentA());
        }
        if (value.parentB() != null) {
            applyGradient(value.parentB());
        }
    }

    private void zeroGradient(Value value) {
        value.zeroGradient();

        if (value.parentA() != null) {
            zeroGradient(value.parentA());
        }
        if (value.parentB() != null) {
            zeroGradient(value.parentB());
        }
    }

    public NetworkProjection networkProjection() {
        List<LayerProjection> layerProjections = new ArrayList<>();
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            layerProjections.add(new LayerProjection(i, layer.neuronProjections()));
        }

        return new NetworkProjection(name, learningRate.data(), layerProjections);
    }
}
