
# Exercise 1: Learn a linear function with PyTorch

## Objective

Se quiere implementar un flujo completo de aprendizaje supervisado para una tarea de regresión con datos sintéticos NO LINEALES ruidosos. 

## Task Formalization

Debe generarse un conjunto de datos mediante una función no lineal con ruido y encapsularlo como un dataset compatible con PyTorch. Luego se debe entrenar el modelo con una función de pérdida adecuada, seleccione el mejor modelo según la pérdida de validación y registre las curvas de pérdida. 

### Task Formalization (Inference)

Se debe implementar un módulo de evaluación que cargue un modelo entrenado, evalúe su desempeño en los conjuntos de entrenamiento, validación y prueba, calcule métricas de regresión (R2, MAE, MSE), y genere gráficos de comparación entre valores reales y predichos. Además, debe guardar las métricas en CSV y como imagen, junto con los gráficos en la carpeta de salida.

### Task Formalization (Training)

Definir un modelo para predecir la salida y construir un proceso de entrenamiento que divida los datos en entrenamiento y validación. Luego se debe entrenar el modelo con una función de pérdida adecuada.

Finalmente se selecciona y guarda el mejor modelo según la pérdida de validación, y se plotea la curva de pérdidas.

## Evaluation metrics

Write your answer here

## Data Considerations

### Dataset description

Datos sintéticos 1D de regresión con ruido, generados como:
y = −3x^2 + 5x + δ

### Data preparation and preprocessing

Generación de x uniforme y añadido de ruido gaussiano.

Se reestructura a tensores columna y se particiona en train/val/test.

### Data augmentation

No se ha implementado.

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

![image](../../outs/exercise_02/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_02/train_regression_plot.png)

![image](../../outs/exercise_02/validation_regression_plot.png)

![image](../../outs/exercise_02/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_02/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_02/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_02/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_02/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Is there overfitting, underfitting or any other issues? 
How can we improve the model?
How this model will generalize to new data?

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

You can include a table stating the chanched parameters and the obtained results after the process.


## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_02` folder.

### Which are the differences you found between previous model and this one?

### Does the model generalizes well to new data?






