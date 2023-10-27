package com.mmc.mlp.model.projection;

import lombok.Value;

import java.util.List;

@Value
public class ModelProjection {

    String name;
    double learningRate;
    long trainedEpochs;
    List<LayerProjection> layers;
}
