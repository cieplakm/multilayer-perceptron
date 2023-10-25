package org.example.perc;


import java.util.Arrays;

public class Tensor {
    public Value[] values;

    public Tensor(double... out) {
        values = Arrays.stream(out)
                .mapToObj(Value::new)
                .toArray(Value[]::new);
    }

    public Tensor(Value[] values) {
        this.values = values;
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (Value value : values) {
            s.append(value.data).append(" ");
        }
        return s.toString();
    }
}
