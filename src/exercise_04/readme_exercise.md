
# Exercise 1: Create a Deep Learning Model for image classification in PyTorch with CIFAR-10 dataset

## Objective

Develop a model that can classify images from CIFAR-10 dataset

Then try a model with convolutional layers
Create an evaluate.py file that evaluates the model and calculates and stores the evaluation metrics including a confusion matrix


Compare this method with previous one (previous exercise)
Whats the effect of data augmentation?

Compare both methods and discuss the differences

## Task Formalization

Objetivo: Problema de clasificacion de imagenes.
Problema: 
La tabla de clases
PL4 CNN (funciona bien) Y en la 5 con la fully vconnected (funciona mal). Como son las imagenes muy pequeñas, si funciona mas o menos bien.
One Shot para que la enquiry al dataset no salga un num (como 6) sino (0 0 0 0 0 1 0 0 0 0).
train, model, evaluate (este es clasificacion --> hacer la matriz de confusion). Metricas de interes: f1 score.
La PL5--> ponemos convolucionales y decimos que funciona peor que CNN.

### Task Formalization (Inference)

Write your answer here
### Task Formalization (Training)

Write your answer here

## Evaluation metrics

Write your answer here

## Data Considerations

### Dataset description

Write your answer here

### Data preparation and preprocessing

Write your answer here

### Data augmentation

Write your answer here

## Model Considerations

Write your answer here

### Suitable Loss Functions

Write your answer here

### Selected Loss Function

Write your answer here

### Possible architectures

Write your answer here

### Last layer activation

Write your answer here

### Other Considerations

Write your answer here

## Training

Write your answer here

### Training hyperparameters

Write your answer here

### Loss function graph

![image](../../outs/exercise_05/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_05/train_regression_plot.png)

![image](../../outs/exercise_05/validation_regression_plot.png)

![image](../../outs/exercise_05/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_05/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_05/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_05/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_05/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Is there overfitting, underfitting or any other issues? 
How can we improve the model?
How this model will generalize to new data?

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

You can include a table stating the chanched parameters and the obtained results after the process.


## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_03` folder.

### Which are the differences you found between previous model and this one?

### Does the model generalizes well to new data?






