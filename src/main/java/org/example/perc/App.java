package org.example.perc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.example.perc.MSE.loss;

class App {

    static class TreningData {
        Tensor value;
        Value target;

        TreningData(Tensor value, Value target) {
            this.value = value;
            this.target = target;
        }
    }

    static TreningData[] treningData = {
            new TreningData(new Tensor(100, 1), Value.of(1)),
            new TreningData(new Tensor(1, 100), Value.of(0)),
            new TreningData(new Tensor(100, 0), Value.of(1)),
            new TreningData(new Tensor(1, 50), Value.of(0)),
            new TreningData(new Tensor(0, 0), Value.of(0)),
            new TreningData(new Tensor(50, 50), Value.of(0)),

    };

    public static void main(String[] args) {
        List<TreningData> trues = IntStream.range(0, 10)
                .mapToObj(i->new TreningData(new Tensor(getLarge(), getSmall()), Value.of(1)))
                .collect(Collectors.toList());
        List<TreningData> falses = IntStream.range(0, 10)
                .mapToObj(i->new TreningData(new Tensor(getSmall(), getLarge()), Value.of(0)))
                .collect(Collectors.toList());

        ArrayList<TreningData> treningData1 = new ArrayList<>(trues);
        treningData1.addAll(falses);
//        treningData = treningData1.toArray(new TreningData[]{});

        LinearLayer linearLayer = new LinearLayer(2, 1);

        Sequential sequential = new Sequential(
                linearLayer
        );

        int epoch = 25;
        System.out.println("-----trening-----");

        for (int i = 0; i < epoch; i++) {
            System.out.println("-----Epoch[" + i + "]------");
            for (TreningData data : treningData) {
                linearLayer.trainNeuron(data.value, data.target);
            }
            Tensor call = sequential.call(new Tensor(new double[]{0, 20}));
            System.out.println("Result:  " + call);
        }

        System.out.println("-----check-----");
        double maxLos = 0;
        for (int i = 0; i <1000000; i++) {
            Tensor call = sequential.call(new Tensor(new double[]{0, 0}));
            double l = loss(call.values[0], Value.of(0)).data;
//            if (loss(call.values[0], Value.of(0)).data > 0.2) {
//                throw new RuntimeException("Chuj");
//            }

            if (l>maxLos){
                maxLos = l;
            }

        }
        System.out.println("MAX LOSS: " + maxLos);
//        System.out.println("Answer: " + call);
    }

    private static double getLarge() {
        Random random = new Random();
        return random.nextDouble(500) + 500;
    }

    private static double getSmall() {
        Random random = new Random();
        return random.nextDouble(500);
    }


}
