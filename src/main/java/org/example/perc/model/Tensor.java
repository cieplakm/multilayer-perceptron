package org.example.perc.model;


import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Tensor {
    private final Value[] values;

    public Tensor(double... out) {
        values = Arrays.stream(out)
                .mapToObj(Value::new)
                .toArray(Value[]::new);
    }

    public Tensor(Value[] values) {
        this.values = values;
    }

    public Value values(int index) {
        return values[index];
    }

    public int size() {
        return values.length;
    }

    @Override
    public String toString() {
        return Stream.of(values)
                .map(Value::toString)
                .collect(Collectors.joining(","));
    }
}
