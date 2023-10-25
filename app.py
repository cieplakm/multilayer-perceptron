import math
import random


class Value:
    def __init__(self, float_number):
        self.data = float_number

    def __add__(sefl, other):
        pass

class Tensor:
    def __init__(self, values):
        self._values = [v if isinstance(v, Value) else Value(v) for v in values]


def main():
    t1 = Tensor([0.1])
    t2 = Tensor([0.5])

    

    print(math.pow(t2-t1, 2))
    print(t2)

if __name__ == "__main__":
    main()