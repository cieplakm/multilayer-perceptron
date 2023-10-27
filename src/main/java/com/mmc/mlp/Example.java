package com.mmc.mlp;

import com.mmc.mlp.model.Networks;
import com.mmc.mlp.model.SequentialNetworkModel;
import com.mmc.mlp.model.Tensor;
import com.mmc.mlp.model.TrainItem;

import static com.mmc.mlp.model.MSE.loss;

public class Example {

    static TrainItem[] trainingData;

    static final Tensor CROSS_FILLED_ANSWER = new Tensor(1, -1);

    static final Tensor NON_CROSS_FILLED_ANSWER = new Tensor(-1, 1);

    static {

        trainingData = new TrainItem[]{
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.7), CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.3), CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.3), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.3), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.3), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.3), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.3), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.7), NON_CROSS_FILLED_ANSWER),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.3), NON_CROSS_FILLED_ANSWER)
        };
    }

    public static void main(String[] args) {
        int layers = 2;
        int inputSize = 4;
        int outputSize = 2;
        double learningRate = 0.1;
        int epochs = 500;
        int[] nextLayersInputSize = new int[]{2};
        boolean loggingTraining = true;

        SequentialNetworkModel newModel = Networks.create("new_model",
                layers,
                inputSize,
                outputSize,
                nextLayersInputSize,
                learningRate);

        newModel.train(epochs, trainingData, loggingTraining);
        Networks.writeToFile(newModel);

        SequentialNetworkModel pretrainedModel = Networks.readFromFile("pretrained_model");

        // test
        Tensor crossedFilled = new Tensor(0.93, 0.15, 0.08, 0.82);
        Tensor nonCrossedFilled = new Tensor(0.45, 0.5, 0.52, 0.4);

        Tensor crossFilledNewModelPrediction = newModel.predict(crossedFilled);
        Tensor nonCrossFilledNewModelPrediction = newModel.predict(nonCrossedFilled);

        Tensor crossFilledPretrainedModelPrediction = pretrainedModel.predict(crossedFilled);
        Tensor nonCrossFilledPretrainedPrediction = pretrainedModel.predict(nonCrossedFilled);

        System.out.println("Is cross filled?");

        System.out.println("---- new model ---");
        System.out.println("Answer: " + interpretation(crossFilledNewModelPrediction));
        System.out.println("Loss: " + loss(crossFilledNewModelPrediction.valueAt(0), CROSS_FILLED_ANSWER.valueAt(0)));
        System.out.println("Answer: " + interpretation(nonCrossFilledNewModelPrediction));
        System.out.println("Loss: " + loss(nonCrossFilledNewModelPrediction.valueAt(1), NON_CROSS_FILLED_ANSWER.valueAt(1)));
        System.out.println("---- pretrained model ---");
        System.out.println("Answer: " + interpretation(crossFilledPretrainedModelPrediction));
        System.out.println("Loss: " + loss(crossFilledPretrainedModelPrediction.valueAt(0), CROSS_FILLED_ANSWER.valueAt(0)));
        System.out.println("Answer: " + interpretation(nonCrossFilledPretrainedPrediction));
        System.out.print("Loss: " + loss(nonCrossFilledPretrainedPrediction.valueAt(1), NON_CROSS_FILLED_ANSWER.valueAt(1)));
    }

    private static Boolean interpretation(Tensor prediction) {
        return prediction.valueAt(0).data() > prediction.valueAt(1).data();
    }


}
