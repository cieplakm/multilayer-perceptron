package org.example.perc.model;


import java.math.BigDecimal;

import static java.math.BigDecimal.valueOf;


public class Value {
    private double data;
    private Value parentA;
    private Value parentB;
    private double gradient = 0;
    private Backward backward;
    private boolean gradientApplicable;

    public static Value of(double v) {
        return new Value(v);
    }

    public Value(double d) {
        this.data = d;
    }

    public Value(double d, Value parentA, Value parentB) {
        this.data = d;
        this.parentA = parentA;
        this.parentB = parentB;
    }

    public static Value ofGradientable(double value) {
        Value value1 = new Value(value);
        value1.gradientApplicable = true;
        return value1;
    }

    public double data() {
        return data;
    }

    public BigDecimal dataBD() {
        return BigDecimal.valueOf(data);
    }

    public Value parentA() {
        return parentA;
    }

    public Value parentB() {
        return parentB;
    }

    public void backward() {
        if (backward != null) {
            backward.back();
        }
    }

    Value add(Value v) {
        Value value = new Value(valueOf(data).add(valueOf(v.data)).doubleValue(), this, v);

        double derivative = 1;

        value.backward = () -> {
            this.gradient = valueOf(gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
            v.gradient = valueOf(v.gradient).add(valueOf(derivative).multiply(valueOf(value.gradient))).doubleValue();
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

    void applyGrad(Value speed) {
        if (gradientApplicable) {
            data = data - gradient * speed.data();
        }
    }

    void zeroGradient() {
        if (gradientApplicable) {
            gradient = 0;
        }
    }

    void gradientOne() {
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
