package com.mmc.mlp.model;

public class MSE {
    public static Value lossLoggable(Value predicted, Value target, boolean log) {
        Value sqr = loss(predicted, target);

        if (log) {
            System.out.println("[y=" + target + "] [pred=" + predicted + "] [err=" + sqr.data() + "]");
        }

        return sqr;
    }

    public static Value loss(Value predicted, Value target) {

        return predicted.subtract(target).sqr();
    }
}
