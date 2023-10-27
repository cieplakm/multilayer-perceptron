package com.mmc.mlp.model;

class MSE {
    static Value loss(Value predicted, Value target, boolean log) {
        Value diff = predicted.subtract(target);
        Value sqr = diff.sqr();

        if (log) {
            System.out.println("[y=" + target + "] [pred=" + predicted + "] [err=" + sqr.data() + "]");
//            System.out.println("[pred="+new DecimalFormat("##.##").format(BigDecimal.valueOf(((predicted.data + 1) * 50))) + "%"+"] [y=" + target + "] [err=" + sqr.data + "]");
        }
        return sqr;
    }
}
