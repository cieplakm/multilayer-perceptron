package com.mmc.mlp.model.projection;

import lombok.Value;

import java.util.List;

@Value
public class ModelProjection {

    String name;
    double learningRate;
    List<LayerProjection> layers;
}
