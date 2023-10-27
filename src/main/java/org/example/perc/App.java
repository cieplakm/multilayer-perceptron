package org.example.perc;

import org.example.perc.model.Networks;
import org.example.perc.model.SequentialNetworkModel;
import org.example.perc.model.Tensor;
import org.example.perc.model.TreningSet;

import java.math.BigDecimal;
import java.text.DecimalFormat;

class App {

    static TreningSet[] treningData;

    static {
        Tensor wrongAnswer = new Tensor(-1, 1);
        Tensor correctAnswer = new Tensor(1, -1);

        treningData = new TreningSet[]{
                new TreningSet(new Tensor(0.7, 0.3, 0.3, 0.7), correctAnswer),
                new TreningSet(new Tensor(0.3, 0.7, 0.7, 0.3), correctAnswer),
                new TreningSet(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.7, 0.7, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.3, 0.7, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.7, 0.3, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.3, 0.3, 0.3), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.7, 0.3, 0.3), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.3, 0.7, 0.3), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.3, 0.3, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.3, 0.3, 0.3), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.7, 0.3, 0.3), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.7, 0.3, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.3, 0.3, 0.7, 0.7), wrongAnswer),
                new TreningSet(new Tensor(0.7, 0.3, 0.7, 0.3), wrongAnswer)
        };
    }

    public static void main(String[] args) {
        SequentialNetworkModel model = Networks.create("cross", 2, 4, 2, new int[]{2}, 0.1);
//        SequentialNetworkModel model = Networks.readFromFile("cross");

        model.train(100_000, treningData, false, 0.0);

        System.out.println(model.predict(treningData[0].getData()));

        Networks.writeToFile(model);
    }
}
