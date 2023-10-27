package com.mmc.mlp.model;

import com.mmc.mlp.model.projection.ModelProjection;

import static com.mmc.mlp.utils.ModelIO.readObjectFromFile;
import static com.mmc.mlp.utils.ModelIO.saveObjectToFile;


public class Networks {

    public static SequentialNetworkModel create(String name, int numOfLayers, int inputSize, int outputSize, int[] layerOutputs, double learningRate) {

        return new SequentialNetworkModel(name, numOfLayers, inputSize, outputSize, layerOutputs, learningRate);
    }

    public static SequentialNetworkModel readFromFile(String modelName) {
        ModelProjection modelProjection = readObjectFromFile(modelName, ModelProjection.class);
        return recreate(modelProjection);
    }

    public static void writeToFile(SequentialNetworkModel model) {
        ModelProjection modelProjection = model.networkProjection();
        saveObjectToFile(modelProjection, modelProjection.getName());
    }

    private static SequentialNetworkModel recreate(ModelProjection projection) {
        Layer[] ls = projection.getLayers()
                .stream()
                .map(layerProjection -> new LinearLayer(layerProjection.getNeurons())).toArray(Layer[]::new);

        return new SequentialNetworkModel(projection.getName(), Value.of(projection.getLearningRate()), projection.getTrainedEpochs(), ls);
    }
}
