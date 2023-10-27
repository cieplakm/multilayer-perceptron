package com.mmc.mlp.model;

class Weights {
    private Value[] weights;

    Weights(Value[] weights) {
        this.weights = weights;
    }

    int size() {
        return weights.length;
    }

    Value weight(int i) {
        return weights[i];
    }
}