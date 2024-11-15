# Artificial Intelligence Engineer Assignment 

This assignment was completed locally and trained on an RTX 2060 GPU using Jupyter Notebook.

## How to run 

Assuming you have installed numpy and tensorflow, you can run each cell in the main.pynb notebook either locally or on Google Colab. Each cell should be executed **strictly** in sequence. The dataset used for the assignment is the popular _MNIST_ dataset, downloaded directly through the notebook from _Kaggle_.

## Assignment tasks

The following text explains the thought process behind each task in the assignment.

### 1. Develop a simple feed forward neural network with three layers with 50 neurons in each layer. 

```python
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  
        Dense(50, activation='relu'),   
        Dense(50, activation='relu'),   
        Dense(50, activation='relu'),   
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model
  ```
A neural network was developed with three _dense_ layers, each using the _ReLU_ activation function, and an output layer with 10 units using _softmax_ activation. There are 10 outputs, corresponding to the 10 classes in the _MNIST_ dataset.

### 3. Split the data in 10 datasets where each dataset consists only one class labels (numbers from 0 to 9) 

After downloading and loading the _MNIST_ dataset, it was split into 10 sub-datasets, each containing instances of only one class (digits 0 to 9). This was done for both the training and test sets.

### 4. Create 3 groups of sub-datasets for each dataset from step 3 by: 

    a. Augmenting each dataset with another 5% rows compared to initial size, where additional 5% rows are randomly selected from other 9 classes of datasets from step 3 (10 new datasets) 
    b. Augmenting each dataset with another 10% rows compared to initial size, where additional 10% rows are randomly selected from other 9 classes of datasets from step 3 (10 new datasets) 
    c. Augmenting each dataset with another 15% rows compared to initial size, where additional 15% rows are randomly selected from other 9 classes of datasets from step 3 (10 new datasets)

  This resulted in a total of 30 datasets, each primarily containing one class but augmented with _5%, 10%,_ or _15%_ additional rows from other classes. An important detail was ensuring that each dataset included _at least some_ of each other class to avoid having datasets without diversity. When calculating the number of additional rows needed `(num_to_add = initial_size * percentage_of_augmentation)`, the rows were distributed across all other classes to include every class in each dataset.
  
### 5. Train all models on datasets from step 4 (30 in total) and use early stopping to determine the number of iterations (and epochs) for each neural network configuration training with each dataset.

In this step, a model was trained for each of the mentioned 30 datasets. Which resulted in having 30 models which were _specialized_ primarily in classifying one class and are less efficient in classifying other classes. While training callback function was defined so that the training proccess stops after 5 epochs where there was no improvement, _val_loss_ was monitored during training to trigger the callback function for early stopping.

```python
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=5,         
                               restore_best_weights=True)
```

Additionally, another callback function tracked each epoch and iteration, and upon completion, stored the number of epochs, iterations, and training time (in seconds) in dictionaries for later analysis.

It was important to _shuffle_ each training set to ensure the model learned effectively, as the augmentation process resulted in datasets with one primary class followed by other classes appended afterward. Shuffling ensured the model could generalize well across all classes in the dataset.
After training, we had three groups of sub-models labeled like this:
```python
models_group_a = [
    models['group_a_class_0'], models['group_a_class_1'], models['group_a_class_2'],
    models['group_a_class_3'], models['group_a_class_4'], models['group_a_class_5'],
    models['group_a_class_6'], models['group_a_class_7'], models['group_a_class_8'],
    models['group_a_class_9']
]

models_group_b = [
    models['group_b_class_0'], models['group_b_class_1'], models['group_b_class_2'],
    models['group_b_class_3'], models['group_b_class_4'], models['group_b_class_5'],
    models['group_b_class_6'], models['group_b_class_7'], models['group_b_class_8'],
    models['group_b_class_9']
]

models_group_c = [
    models['group_c_class_0'], models['group_c_class_1'], models['group_c_class_2'],
    models['group_c_class_3'], models['group_c_class_4'], models['group_c_class_5'],
    models['group_c_class_6'], models['group_c_class_7'], models['group_c_class_8'],
    models['group_c_class_9']
```

### 6. Develop three ensemble models based on step 4 datasets (4.a, 4.b and 4c model) to test how mixture of additional data impacts performance (5%, 10% and 15%) and implement voting or other ensemble method to obtain final accuracy. 

Developing ensemble models
In this step, _three_ ensemble models are developed and they are based on the models trained in the previous step.
- First variation of ensemble model uses _max voting method_ where the final prediction is the output of the submodel which has the _highest_ confidence in its prediction.
- Second variation of ensemble model are based on _averaging_ the prediction confidences from all of the submodels in the ensemble. We take an average confidence for each class from all submodels and as a final output we take the _maximum average confidence_.
- Third variation of ensemble model is based on using the knowledge of each submodel's _"specialty class"_, for example: submodel called _"group_a_class_0"_ is primarily trained on a class 0 with a small percentage of other classes. So, using this knowledge, about this submodel, we _weigh_ its confidence about its prediction more than other model's prediction confidences.

The standard voting method was not used because of the nature of the problem. Each submodel is specialized for classifing only one class with solid accuracy, so the standard voting method would result with 1 good submodel's prediction versus 9 other submodel's predictions, where other 9 submodels are not properly trained for that class, which would most likely result in bad final prediction.

### 7. Measure the total learning time for each of three models (training of 10 sub-models), and calculate the total number of iterations per three models (sum of iterations of sub-models) 

For this step, the stored data from _Task 4_ was used to calculate the total training time for each of three models. In the following table, row _group_ contains three groups: group a, group b and group c. _Group a_ is a group of sub-models where each sub-model is trained on augmented sub-dataset with additional 5% rows randomly selected from other 9 classes. Group B and C are groups trained on augmented sub-datasets with additional 10% and 15% rows, respectively.
```
         total_training_time  total_iterations
group                                         
group_a           112.467002           18418.0
group_b           119.373199           20035.0
group_c           119.006512           19181.0
```
From the table, it can be observed that it took more time and iterations to traing group b of submodels than to traing group a submodels. But it can also be observed that the training time and iterations for training group c submodels was less than the training time and iterations of group b submodels. Which is interesting, this could happen due to various factors. It could mean that the subdatasets used for group c submodel training was better and more _diverse_ and it provided better ground for faster convergence and that this diversity helped the models generalize better. Also, because it was trained locally and because the training time and number of iterations were fairly similar, it could simply mean that the hardware(GPU) was not utilisied as efficiently as in the group of submodels trained on a smaller dataset.

### 8. Test three ensemble models with test data. 
```
Max vote variant
Ensemble accuracy for Group A models: 91.92%
Ensemble accuracy for Group B models: 93.67%
Ensemble accuracy for Group C models: 93.87%

Averaging confidences variant
Ensemble accuracy (averaging) for Group A models: 89.37%
Ensemble accuracy (averaging) for Group B models: 91.64%
Ensemble accuracy (averaging) for Group C models: 92.02%

Weighted confidences variant
Ensemble accuracy for Group A models: 91.93%
Ensemble accuracy for Group B models: 93.59%
Ensemble accuracy for Group C models: 94.02%
```
### 9. Plot the dependence of accuracy (x-axis) and training time (y-axis) on one scatter plot, and dependence of accuracy (x-axis) and number of iterations (y-axis) on second scatter plot. 

![scatter-plot](https://github.com/user-attachments/assets/8f6fc122-0c06-402f-8845-3be208987a01)

### 10. Comment and discuss the final results in analysis posted in GitHuB readme.

The results demonstrate how the ensemble accuracy varies across the three models (Groups A, B, and C) and across three ensemble methods (Max Vote, Averaging Confidences, and Weighted Confidences):
- Max Vote Variant:
    * Accuracy increases from Group A (91.92%) to Group C (93.87%), with Group B performing slightly below Group C.
    * The increasing trend indicates that higher augmentation percentages (Group C with 15%) provide better generalization, as the ensemble can better predict test data by leveraging more diverse individual sub-models.

- Averaging Confidences Variant:
    * Accuracy follows a similar upward trend but is consistently lower compared to the Max Vote method for all groups.
    * This suggests that averaging model confidences may dilute the effectiveness of strong predictions, especially when sub-models are highly confident in correct classes.

- Weighted Confidences Variant:
    * This method achieves the highest accuracy overall, with Group C (94.02%) performing the best.
    * The performance increase highlights that assigning weights to confidences helps emphasize the contribution of better-performing sub-models, leading to improved predictions.
 
Group C consistently outperforms Groups A and B across all ensemble methods, showcasing that augmenting with 15% additional rows provides the most robust sub-models.
Group B slightly outperforms Group A in all methods, confirming that a higher augmentation percentage (10% vs. 5%) positively impacts ensemble performance.


- Overall Trends:
    * Weighted Confidences > Max Vote > Averaging Confidences in terms of accuracy.
    * As the augmentation percentage increases, ensemble accuracy improves, with Group C achieving the best results in all cases.
 
Based on the previously shown scatter plot, it can be observed that the two variants of ensemble models _Max voting_ and _Weighted_ variants have similar accuracy results where _Weighted_ variant has shown a slight edge over _Max voting_ variant. The _Averaging_ variant has consistently shown worse results in terms of accuracy across all groups (group a, group b and group c). What is also interesting and can be observed from the given scatter plot is that bigger number of iterations and more training time did _not_ necessarily mean better results in terms of accuracy, which can be observed by the number of iterations and training time on group b and group c, where group b has smaller dataset (sub-datasets with 10% augmentation versus 15% augmentation in group c) to train on but has similar training time and number of iterations to the training time and number of iterations while training group c ensemble models where the dataset used for it's training has more data. This information could give as an insight on what is the optimal ammount of augmentation of sub-datasets to achieve optimal accuracy _and_ optimal efficiency in terms of resource usage (training time and number of iterations).
