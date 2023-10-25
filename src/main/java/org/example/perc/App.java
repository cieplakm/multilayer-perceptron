package org.example.perc;

import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.example.perc.MSE.loss;

class App {

    static TreningData[] treningData = {
            new TreningData(new Tensor(3.31, 1.58, 0.66, 0.97, 0.3), Value.of(1)),
//            new TreningData(new Tensor(2, 5), Value.of(-1)),
    };

    public static void main(String[] args) {
//        treningData = extendedTreningData().toArray(new TreningData[]{});

        LinearLayer linearLayer = new LinearLayer(5, 2);
        LinearLayer linearLayer2 = new LinearLayer(2, 4);
        LinearLayer linearLayer3 = new LinearLayer(4, 1);
        Sequential sequential = new Sequential(linearLayer, linearLayer2, linearLayer3);

        int epoch = 100;
        for (int i = 0; i < epoch; i++) {
            System.out.println("-----Epoch[" + i + "]------");
            for (TreningData data : treningData) {
                Tensor prediction = sequential.predict(data.value);
                for (Value predVal : prediction.values) {
                    Value loss = loss(predVal, data.target);
                    loss.gradient = 1;
                    backpropagate(loss);
                    applyGradient(loss);
                }
            }
        }

        checkNTimes(sequential);

//        Tensor call = sequential.predict(new Tensor(6, 4, 1.5, 1, 0.9));
//        System.out.println("Answer: " + call);
//        System.out.println("Answer: " +new DecimalFormat("##.##").format(BigDecimal.valueOf(((call.values[0].data+1)*50))) + "%");
    }

    private static void checkNTimes(Sequential sequential) {
        System.out.println("-----check-----");
        double maxLos = 0;
        double sum = 0;

        int times = 100;
        for (int i = 0; i < times; i++) {
            Tensor call = sequential.predict(new Tensor(6, 4, 1.5, 1, 0.9));
            double l = loss(call.values[0], Value.of(1)).data;
            sum += l;
            if (Math.abs(l) > Math.abs(maxLos)) {
                maxLos = l;
            }
        }
        System.out.println("MAX LOSS: " + maxLos);
        System.out.println("AVG LOSS: " + sum / times);
    }

    private static void applyGradient(Value value) {
        value.applyGrad();

        if (value.parentA != null) {
            applyGradient(value.parentA);
        }
        if (value.parentB != null) {
            applyGradient(value.parentB);
        }
    }

    private static void backpropagate(Value value) {
        value.backward.back();

        if (value.parentA != null && value.parentA.backward != null) {
            backpropagate(value.parentA);
        }
        if (value.parentB != null && value.parentB.backward != null) {
            backpropagate(value.parentB);
        }
    }

    private static double getLarge() {
        Random random = new Random();
        return random.nextDouble(500) + 500;
    }

    private static double getSmall() {
        Random random = new Random();
        return random.nextDouble(500);
    }

    private static ArrayList<TreningData> extendedTreningData() {
        List<TreningData> trues = IntStream.range(0, 100)
                .mapToObj(i -> new TreningData(new Tensor(getLarge(), getSmall()), Value.of(1)))
                .collect(Collectors.toList());
        List<TreningData> falses = IntStream.range(0, 100)
                .mapToObj(i -> new TreningData(new Tensor(getSmall(), getLarge()), Value.of(-1)))
                .collect(Collectors.toList());

        ArrayList<TreningData> treningData1 = new ArrayList<>(trues);
        treningData1.addAll(falses);
        return treningData1;
    }

    static class TreningData {
        Tensor value;
        Value target;

        TreningData(Tensor value, Value target) {
            this.value = value;
            this.target = target;
        }
    }
}
