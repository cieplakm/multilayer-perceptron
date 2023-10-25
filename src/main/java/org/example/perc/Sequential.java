package org.example.perc;

class Sequential {
    private final Layer[] layers;

    public Sequential(Layer... layers) {
        this.layers = layers;
    }

    Tensor call(Tensor data) {

        for (Layer layer : layers) {
            data = layer.call(data);
        }

        return data;
    }
}
