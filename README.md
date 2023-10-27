# Multilayer perceptron model example

## Usage

### Training data
```java
Tensor wrongAnswer = new Tensor(-1, 1);
Tensor correctAnswer = new Tensor(1, -1);
        
TrainItem[] trainingData = = new TrainItem[] {
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.7), correctAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.3), correctAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.7, 0.3, 0.3), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.7, 0.3, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.3, 0.3, 0.7, 0.7), wrongAnswer),
                new TrainItem(new Tensor(0.7, 0.3, 0.7, 0.3), wrongAnswer)
        };
```
### Train new model
```java
String modelName= "some_model";
int layers = 2;
int inputSize = 4;
int outputSize = 2;
double learningRate = 0.1;
int epoch = 10_000;
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
        
Tensor toCheck = new Tensor(1, 0.1, 1, 0.5);

Tensor prediction = modelmodel.predict(toCheck);

double probability = prediction.valueAt(0);
probability > 0 // => probable true
probability < 0 // => probable false

```