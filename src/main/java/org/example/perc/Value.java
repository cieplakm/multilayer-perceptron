package org.example.perc;


public class Value {
    public double data;
    Value parent;
    Value parent2;
    double gradient = 0;
    Backward backward;

    public static Value of(int v) {
        return new Value(v);
    }

    @Override
    public String toString() {
        return String.valueOf(data);
    }

    Value(double d) {
        this.data = d;
    }

    Value(double d, Value parent, Value parent2) {
        this.data = d;
        this.parent = parent;
        this.parent2 = parent2;
    }

    Value add(Value v) {
        Value value = new Value(data + v.data, this, v);
        double dev = 1;

        value.backward = () -> {
            this.gradient += dev * value.gradient;
            v.gradient += dev * value.gradient;
        };
        return value;
    }

    Value subtract(Value v) {
        Value value = new Value(data - v.data, this, v);
        double dev = 1;
        value.backward = () -> {
            this.gradient += dev * value.gradient;
            v.gradient += dev * value.gradient;
        };
        return value;
    }

    Value multiply(Value v) {
        Value value = new Value(data * v.data, this, v);

        double dev = 1;
        value.backward = () -> {
            this.gradient += dev * value.gradient;
            v.gradient += dev * value.gradient;
        };
        return value;
    }

    Value sqr() {
        Value value = new Value(Math.pow(data, 2), this, null);

        double dev = 2 * data;

        value.backward = () -> {
            this.gradient += dev * value.gradient;
        };
        return value;
    }

    Value tanh() {
        double tanh = Math.tanh(data);//;
        Value value = new Value(tanh, this, null);

        double dev = 1 - (Math.pow(tanh, 2));

        value.backward = () -> {
            this.gradient += dev * value.gradient;
        };
        return value;
    }

    Value sigmoid() {
        Value value = new Value(1.0 / (1.0 + Math.exp(-data)));
        return value;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void applyGrad(double speed) {
        data = data - gradient * speed;
        gradient = 0;
    }

    interface Backward {
        void back();
    }
}
