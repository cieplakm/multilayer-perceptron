package org.example.perc;

public class LogLoss {

    public static void main(String[] args) {
        double[] actualLabels = {1, 0, 1};
        double[] predictedProbabilities = {0.8, 0.2, 0.6};

        double loss = calculateLogLoss(actualLabels, predictedProbabilities);
        System.out.println("Log Loss: " + loss);
    }

    public static double calculateLogLoss(double[] actualLabels, double[] predictedProbabilities) {
        if (actualLabels.length != predictedProbabilities.length) {
            throw new IllegalArgumentException("Input arrays must have the same length");
        }

        double logLoss = 0.0;
        for (int i = 0; i < actualLabels.length; i++) {
            double prob = Math.min(Math.max(predictedProbabilities[i], 1e-15), 1 - 1e-15);
            logLoss += actualLabels[i] * Math.log(prob) + (1 - actualLabels[i]) * Math.log(1 - prob);
        }

        return -logLoss / actualLabels.length;
    }
}
