package com.mmc.mlp.model.projection;

import lombok.Value;

import java.util.List;

@Value
public class LayerProjection {

    int id;
    List<NeuronProjection> neurons;
}
