package com.mmc.mlp;

import com.mmc.mlp.model.Networks;
import com.mmc.mlp.model.SequentialNetworkModel;
import com.mmc.mlp.model.Tensor;
import com.mmc.mlp.model.TrainItem;

class Example {

    static TrainItem[] trainingData;

    static {
        Tensor wrongAnswer = new Tensor(-1, 1);
        Tensor correctAnswer = new Tensor(1, -1);

        trainingData = new TrainItem[]{
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.7), correctAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.3), correctAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.3), wrongAnswer)
        };
    }

    public static void main(String[] args) {
        String modelName = "some_model";
        int layers = 2;
        int inputSize = 4;
        int outputSize = 1;
        double learningRate = 0.1;
        int epoch = 10_000;
        int[] nextLayersInputSize = new int[]{2};
        boolean loggingTrainig = true;

        SequentialNetworkModel model = Networks.create(modelName,
                layers,
                inputSize,
                outputSize,
                nextLayersInputSize,
                learningRate);

        model.train(epoch, trainingData, loggingTrainig);

        Networks.writeToFile(model);
    }
}
