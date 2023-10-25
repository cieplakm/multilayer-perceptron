package org.example.perc;

class Sequential {
    private final Layer[] layers;

    public Sequential(Layer... layers) {
        this.layers = layers;
    }

    Tensor predict(Tensor data) {

        for (Layer layer : layers) {
            Tensor call = layer.call(data);
            data = call;
        }

        return data;
    }
}
