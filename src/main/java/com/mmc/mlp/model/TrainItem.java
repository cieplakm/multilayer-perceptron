package com.mmc.mlp.model;

import lombok.Value;

@Value
public class TrainItem {

    Tensor data;
    Tensor target;
}
