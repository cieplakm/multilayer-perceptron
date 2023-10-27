# Multilayer perceptron model example

### Problem
Try to recognize when cross (0,3;1,2) fields are filled.
```
+-----+-----+
|     |     | 
|  0  |  1  | 
|     |     | 
+-----+-----+
|     |     | 
|  2  |  3  | 
|     |     | 
+-----+-----+
```


### Training data
```java
Tensor crossFilledAnswer = new Tensor(-1, 1);
Tensor nonCrossFilledAnswer = new Tensor(1, -1);
        
TrainItem[] trainingData = = new TrainItem[] {
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.7), crossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.3), crossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.3), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.3), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.3), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.3), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.3), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.7), nonCrossFilledAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.3), nonCrossFilledAnswer)
        };
```
### Train new model


```java
String modelName= "some_model";
int layers = 2;
int inputSize = 4;
int outputSize = 2;
double learningRate = 0.1;
int epoch = 1_000;
int[] nextLayersInputSize = new int[]{2}; //lenght: (layers-1)
boolean loggingTrainig = true;

SequentialNetworkModel model = Networks.create(modelName, 
        layers, 
        inputSize, 
        outputSize, 
        nextLayersInputSize, 
        learningRate);

model.train(epoch, trainingData, loggingTrainig);

Networks.writeToFile(model);
```

### Train existing model
```java
String = "some_model";
boolean loggingTrainig = false;

SequentialNetworkModel model = Networks.readFromFile(modelName);

model.train(epoch, trainingData, loggingTrainig);

Networks.writeToFile(model);
```

### Usage
```java
SequentialNetworkModel model = ...
        
Tensor toCheck = new Tensor(0.93, 0.15, 0.08, 0.82); // 0,4 - filled, 1,3 - not filled 

Tensor prediction = modelmodel.predict(toCheck);

double probability = prediction.valueAt(0);
probability > 0 // => probable true
probability < 0 // => probable false

```