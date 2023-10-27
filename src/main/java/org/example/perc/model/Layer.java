package org.example.perc.model;

import org.example.perc.projection.NeuronProjection;

import java.util.List;

interface Layer {
    Tensor call(Tensor tensor);

    List<NeuronProjection> neuronProjections();
}
