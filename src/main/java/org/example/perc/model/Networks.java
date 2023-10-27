package org.example.perc.model;

import org.example.perc.projection.NetworkProjection;

import static org.example.perc.ObjectToFile.readObjectFromFile;
import static org.example.perc.ObjectToFile.saveObjectToFile;

public class Networks {

    public static SequentialNetworkModel create(String name, int numOfLayers, int inputSize, int outputSize, int[] layerOutputs, double learningRate) {

        return new SequentialNetworkModel(name, numOfLayers, inputSize, outputSize, layerOutputs, learningRate);
    }

    public static SequentialNetworkModel readFromFile(String modelName) {
        NetworkProjection networkProjection = readObjectFromFile(modelName, NetworkProjection.class);
        return recreate(networkProjection);
    }

    public static void writeToFile(SequentialNetworkModel model) {
        NetworkProjection networkProjection = model.networkProjection();
        saveObjectToFile(networkProjection, networkProjection.getName());
    }

    static SequentialNetworkModel recreate(NetworkProjection projection) {
        Layer[] ls = projection.getLayers()
                .stream()
                .map(layerProjection -> new LinearLayer(layerProjection.getNeurons())).toArray(Layer[]::new);

        return new SequentialNetworkModel(projection.getName(), Value.of(projection.getLearningRate()), ls);
    }
}
