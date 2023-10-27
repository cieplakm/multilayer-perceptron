package com.mmc.mlp.model;

import com.mmc.mlp.model.projection.NeuronProjection;

import java.util.List;

interface Layer {
    Tensor call(Tensor tensor);

    List<NeuronProjection> neuronProjections();
}
