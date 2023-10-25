package org.example.perc;


import java.util.Arrays;

public class Tensor {
    public Value[] values;

//    public Tensor(double[] out) {
//        values = Arrays.stream(out)
//                .mapToObj(v -> new Value(v))
//                .toArray(Value[]::new);
//    }

    public Tensor(double... out) {
        values = Arrays.stream(out)
                .mapToObj(v -> new Value(v))
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
    //        void backpropagate() {
//            values[values.length - 1].gradient = 1;
//
//            for (Value value : values) {
//                if (value.backward != null) {
//                    recVal(value);
//                }
//            }
//        }
//
//        void applyGrad(double speed) {
//            for (Value value : values) {
//                if (value != null) {
//                    gradRec(value, speed);
//                }
//            }
//        }

    void gradRec(Value value, double speed) {
        value.applyGrad(speed);
        if (value.parent != null) {
            gradRec(value.parent, speed);
        }
        if (value.parent2 != null) {
            gradRec(value.parent2, speed);
        }
    }

    public void recVal(Value value) {
        if (value.backward != null) {
            value.backward.back();
        }
        if (value.parent != null) {
            recVal(value.parent);
        }
        if (value.parent2 != null) {
            recVal(value.parent2);
        }
    }
}
