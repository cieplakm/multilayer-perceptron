package org.example.perc;


import java.math.BigDecimal;

import static java.math.BigDecimal.*;
import static java.math.BigDecimal.valueOf;

public class Value {
    double data;
    Value parentA;
    Value parentB;
    double gradient = 0;
    Backward backward;

    boolean gradientApplicable;

    public static Value of(int v) {
        return new Value(v);
    }

    Value(double d) {
        this.data = d;
    }

    Value(double d, Value parentA, Value parentB) {
        this.data = d;
        this.parentA = parentA;
        this.parentB = parentB;
    }

    public static Value ofGradientable(double value) {
        Value value1 = new Value(value);
        value1.gradientApplicable = true;
        return value1;
    }

    Value add(Value v) {
        Value value = new Value(valueOf(data).add(valueOf(v.data)).doubleValue(), this, v);

        double derivative = 1;

        value.backward = () -> {
            this.gradient = valueOf(gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
            v.gradient =  valueOf(v.gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
        };

        return value;
    }

    Value subtract(Value v) {
        Value value = new Value(valueOf(data).subtract(valueOf(v.data)).doubleValue(), this, v);

        double derivative = 1;

        value.backward = () -> {
            this.gradient = valueOf(gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
            v.gradient =  valueOf(v.gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
        };
        return value;
    }

    Value multiply(Value v) {
        Value value = new Value(valueOf(data).multiply(valueOf(v.data)).doubleValue() , this, v);

        value.backward = () -> {
            this.gradient = valueOf(this.gradient).add(valueOf(v.data).multiply(valueOf(value.gradient))).doubleValue();
            v.gradient = valueOf(v.gradient).add(valueOf(this.data).multiply(valueOf(value.gradient))).doubleValue();
        };

        return value;
    }

    Value sqr() {
        Value value = new Value(BigDecimal.valueOf(data).pow(2).doubleValue(), this, null);

        double derivative = valueOf(data).multiply(valueOf(2.0)).doubleValue();

        value.backward = () -> {
            this.gradient = valueOf(this.gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
        };

        return value;
    }

    Value tanh() {
        double tanh = Math.tanh(data);
        Value value = new Value(tanh, this, null);

        double derivative = 1 - (Math.pow(tanh, 2));

        value.backward = () -> {
            this.gradient = valueOf(this.gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
        };

        return value;
    }

//    Value sigmoid() {
//        double sigmoidValue = sigmoid(data);
//        Value value = new Value(sigmoidValue, this, null);
//
//        double derivative = sigmoidValue * (1 - sigmoidValue);
//
//        value.backward = () -> {
//            this.gradient += derivative * value.gradient;
//        };
//
//        return value;
//    }

    public void applyGrad() {
        if (gradientApplicable) {
            data = data - gradient * 0.01;
        }
    }

    public void zeroGrad() {
        if (gradientApplicable) {
            gradient = 0;
        }
    }

    public void gradinetOne() {
        gradient = 1;
    }

    interface Backward {

        void back();
    }


    @Override
    public String toString() {
        return String.valueOf(data);
    }
}
