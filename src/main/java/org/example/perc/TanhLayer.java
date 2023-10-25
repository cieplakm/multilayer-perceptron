package org.example.perc;

import java.util.Arrays;

class TanhLayer implements Layer {
    @Override
    public Tensor call(Tensor tensor) {
        Value[] array = Arrays.stream(tensor.values)
                .map(Value::tanh)
                .toArray(Value[]::new);
        return new Tensor(array);
    }
}
