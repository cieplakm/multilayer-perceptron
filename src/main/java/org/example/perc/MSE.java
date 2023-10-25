package org.example.perc;

import java.math.BigDecimal;
import java.text.DecimalFormat;

class MSE {
    static Value loss(Value predicted, Value target) {
        Value prediction = predicted;
        Value diff = prediction.subtract(target);
        Value sqr = diff.sqr();

        System.out.println("[y=" + target + "]" + "[pred="+predicted+"]" + "[err=" + sqr.data + "]");
//        System.out.println("[pred="+new DecimalFormat("##.##").format(BigDecimal.valueOf(((predicted.data + 1) * 50))) + "%"+"] [y=" + target + "] [err=" + sqr.data + "]");
        return sqr;
    }
}
