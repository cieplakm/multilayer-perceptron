package com.mmc.mlp.model;

import com.mmc.mlp.model.projection.LayerProjection;
import com.mmc.mlp.model.projection.ModelProjection;

import java.util.ArrayList;
import java.util.List;

public class SequentialNetworkModel {
    private final String name;
    private final Value learningRate;
    private final Layer[] layers;
    private long trainedEpochs;


    SequentialNetworkModel(String name, Value learningRate, long trainedEpochs, Layer... layers) {
        this.name = name;
        this.learningRate = learningRate;
        this.trainedEpochs = trainedEpochs;
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
            this.layers[layer] = new LinearLayer(inSize(layer, inputSize, layerOutputs), outSize(numOfLayers, layer, outputSize, layerOutputs));
        }
    }

    public Tensor predict(Tensor data) {
        for (Layer layer : layers) {
            data = layer.call(data);
        }

        return data;
    }

    public void train(int epoch, TrainItem[] trainItems, boolean logging) {
        long t1 = System.currentTimeMillis();
        for (int i = 0; i < epoch; i++) {
            for (TrainItem data : trainItems) {
                Tensor prediction = predict(data.getData());
                for (int k = 0; k < prediction.size(); k++) {
                    Value loss = MSE.lossLoggable(prediction.valueAt(k), data.getTarget().valueAt(k), logging);
                    loss.gradientOne();
                    backPropagate(loss);
                    applyGradient(loss);
                    zeroGradient(loss);
                }
            }
        }

        trainedEpochs += epoch;

        long t2 = System.currentTimeMillis();
        System.out.println(String.format("Training %s epochs is done in %sms", epoch, t2 - t1));
    }

    public ModelProjection networkProjection() {
        List<LayerProjection> layerProjections = new ArrayList<>();
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            layerProjections.add(new LayerProjection(i, layer.neuronProjections()));
        }

        return new ModelProjection(name, learningRate.data(), trainedEpochs, layerProjections);
    }

    private int inSize(int layer, int inputSize, int[] doubles) {
        if (layer == 0) {
            return inputSize;
        }

        return doubles[layer - 1];
    }

    private static int outSize(int numOfLayers, int layer, int outputSize, int[] doubles) {
        if (layer == numOfLayers - 1) {
            return outputSize;
        }

        return doubles[layer];
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
}
