package org.example.perc.projection;

import lombok.Value;

import java.util.List;

@Value
public class NeuronProjection {
    int id;
    double bias;
    List<WeightProjection> weights;
}
