package org.example.perc.projection;

import lombok.Value;

import java.util.List;

@Value
public class NetworkProjection {

    String name;
    double learningRate;
    List<LayerProjection> layers;
}
