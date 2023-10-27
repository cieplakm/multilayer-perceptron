package org.example.perc.projection;

import lombok.Value;

import java.util.List;

@Value
public class LayerProjection {

    int id;
    List<NeuronProjection> neurons;
}
