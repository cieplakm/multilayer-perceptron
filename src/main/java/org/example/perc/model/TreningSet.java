package org.example.perc.model;

import lombok.Value;

@Value
public class TreningSet {

    Tensor data;
    Tensor target;
}
