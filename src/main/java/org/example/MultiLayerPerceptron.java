package org.example;
public class MultiLayerPerceptron {
    private double[][] inputLayerWeights;
    private double[][] hiddenLayerWeights;
    private double[] outputLayerWeights;

    public MultiLayerPerceptron(int inputSize, int hiddenSize, int outputSize) {
        inputLayerWeights = initializeWeights(inputSize, hiddenSize);
        hiddenLayerWeights = initializeWeights(hiddenSize, outputSize);
        outputLayerWeights = new double[outputSize];

        initializeWeights(outputSize, 1);
    }

    private double[][] initializeWeights(int numRows, int numCols) {
        double[][] weights = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                // Initialize weights with random values between -1 and 1
                weights[i][j] = Math.random() * 2 - 1;
            }
        }
        return weights;
    }

    private double sigmoid(double x) {
        // Sigmoid activation function
        return 1 / (1 + Math.exp(-x));
    }

    public double[] forward(double[] input) {
        double[] hiddenActivations = calculateActivations(input, inputLayerWeights);
        double[] outputActivations = calculateActivations(hiddenActivations, hiddenLayerWeights);

        // Output layer is just a single neuron in this example
        double output = sigmoid(outputActivations[0] + outputLayerWeights[0]);

        return new double[]{output};
    }

    private double[] calculateActivations(double[] input, double[][] weights) {
        double[] activations = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double sum = 0;
            for (int j = 0; j < Math.min(input.length, weights[i].length); j++) {
                sum += input[j] * weights[i][j];
            }
            activations[i] = sigmoid(sum);
        }
        return activations;
    }


    public static void main(String[] args) {
        int inputSize = 3;
        int hiddenSize = 4;
        int outputSize = 2;

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(inputSize, hiddenSize, outputSize);

        // Example input
        double[] input = {0.5, 0.3, 0.9};

        // Forward pass
        double[] output = mlp.forward(input);

        // Display the output
        System.out.println("Output:");
        for (double value : output) {
            System.out.println(value);
        }
    }
}
