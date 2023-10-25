package org.example.perc;

class MSE {
    static Value loss(Value predicted, Value target) {
        Value prediction = predicted;
        Value diff = target.subtract(prediction);
        Value sqr = diff.sqr();

        System.out.println("[loss=" + sqr.data + "] [y_pred="+predicted+"] [y=" + target + "]");
        return sqr;
    }
}
