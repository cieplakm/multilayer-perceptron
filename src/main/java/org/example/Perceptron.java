package org.example;

import java.util.Random;

public class Perceptron {

    private double weight;
    private double bias;
    private double learningRate;

    public Perceptron(double learningRate) {
        this.bias = new Random().nextDouble();
        this.weight = new Random().nextDouble();
        this.learningRate = learningRate;
    }

    public double predict(double input) {
        return sigmoid(this.weight * input + this.bias);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double leakyRelu(double x) {
        return Math.max(0, x);
    }

    public void train(double input, double target) {
        double prediction = predict(input);
        double error = target - prediction;

        weight += error * input;
        bias += error;
    }

    public static void main(String[] args) {
        Perceptron perceptron = new Perceptron(0.1);

        for (int i = 0; i < 1000; i++) {
//            perceptron.train(1, 1);
//            perceptron.train(2, 1);
//            perceptron.train(3, 1);
//            perceptron.train(4, 1);
            perceptron.train(5, 1);

                perceptron.train(7, 0);
//            perceptron.train(7, 0);
//            perceptron.train(8, 0);
//            perceptron.train(9, 0);
//            perceptron.train(10, 0);
        }
        double prediction1 = perceptron.predict(5.5);


        System.out.println("B: " + perceptron.bias + " " + "W: " + perceptron.weight);
        System.out.println("Prediction for 5: " + prediction1);
    }
}
