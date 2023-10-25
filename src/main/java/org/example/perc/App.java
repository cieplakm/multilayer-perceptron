package org.example.perc;

import static org.example.perc.MSE.loss;

class App {

    static TreningData[] treningData = {
            new TreningData(new Tensor(3.305, 1.578, 0.66, 0.978, 0.295), Value.of(1)),
            new TreningData(new Tensor(1.043, 0.531, 0.202, 0.102, 0.08), Value.of(-1)),
            new TreningData(new Tensor(2.381, 1.554, 1.03, 2.17, 0.726), Value.of(-1)),
    };

    public static void main(String[] args) {
        LinearLayer linearLayer = new LinearLayer(5, 2);
        LinearLayer linearLayer2 = new LinearLayer(2, 4);
        LinearLayer linearLayer3 = new LinearLayer(4, 1);
        Sequential sequential = new Sequential(linearLayer, linearLayer2, linearLayer3);

        int epoch = 10000;
        for (int i = 0; i < epoch; i++) {
            for (TreningData data : treningData) {
                Tensor prediction = sequential.predict(data.value);
                Value loss = loss(prediction.values[0], data.target);
                loss.gradinetOne();
                backpropagate(loss);
                applyGradient(loss);
                zeroGradient(loss);
            }
        }

        checkNTimes(sequential);

//        Tensor call = sequential.predict(new Tensor(3.31, 1.58, 0.66, 0.97, 0.3));
//        System.out.println("Answer: " + call);
//        System.out.println("Answer: " + new DecimalFormat("##.##").format(BigDecimal.valueOf(((call.values[0].data + 1) * 50))) + "%");
    }


    private static void checkNTimes(Sequential sequential) {
        System.out.println("----- TESTING N TIMES -----");


        Tensor data = new Tensor(50.5, 20.1, 3, 15, 1);


        double maxLos = 0;
        double sum = 0;

        int times = 5;
        for (int i = 0; i < times; i++) {
            Tensor call = sequential.predict(data);
            double l = loss(call.values[0], Value.of(1)).data;
            sum += l;
            if (Math.abs(l) > Math.abs(maxLos)) {
                maxLos = l;
            }
        }
//        System.out.println("MAX LOSS: " + maxLos);
//        System.out.println("AVG LOSS: " + sum / times);
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

    private static void zeroGradient(Value value) {
        value.zeroGrad();

        if (value.parentA != null) {
            zeroGradient(value.parentA);
        }
        if (value.parentB != null) {
            zeroGradient(value.parentB);
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

    static class TreningData {
        Tensor value;
        Value target;

        TreningData(Tensor value, Value target) {
            this.value = value;
            this.target = target;
        }
    }
}
