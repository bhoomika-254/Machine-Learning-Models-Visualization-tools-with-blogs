# SGD-Visualiser

# Random Forest

- Random Forest is a bagging technique, which stands for bootstrapped aggregation. Bootstrapping involves random sampling of data; for example, if we have a dataset with 2000 rows, we might divide it into subsets of 500 rows each and send these subsets to multiple models.

- Bootstrapping can be done on rows, columns, or both rows and columns. There are two options for bootstrapping: with replacement or without replacement. Row sampling with replacement means a row can be selected more than once, resulting in duplicate rows. Without replacement means each row is picked only once for a single decision tree. The same options apply to column sampling, with the possibility of combining both row and column sampling in various permutations.

- Aggregation in Random Forest means querying all models and determining the final output based on the majority vote for classification tasks (e.g., if Model 1 and Model 2 give an output of 1, and Model 3 gives an output of 0, the final output is 1). For regression tasks, the final output is the mean of all model outputs.

- In a Random Forest, we train multiple decision trees simultaneously.

- Typically, there's an inverse relationship between bias and variance, making it challenging to achieve both low bias and low variance in a machine learning model. This balance is crucial to avoid overfitting and underfitting. High bias, low variance models, such as Linear Regression and Logistic Regression, tend to underfit, while low bias, high variance models, such as SVM, KNN, and fully grown decision trees, may overfit.

- Random Forest achieves a balance by reducing variance while maintaining low bias. This is accomplished through the process of bagging and random feature selection. In Random Forest, subsets of the dataset are sent to individual decision trees. Each tree may receive different random subsets, including different rows and columns, with or without replacement. This means that not all trees will receive the same noisy data points or outliers, which helps to average out errors and reduce variance.

- For example, if we introduce noise or outliers by randomly changing 100 rows in the dataset, a fully grown decision tree might overfit to this noisy data, performing well on the training data but poorly on the test data. However, in a Random Forest, where multiple decision trees are trained on different subsets of the data, it's unlikely that all trees will overfit to the same noisy points. This randomization helps to capture the true nature of the dataset, thus improving generalization and reducing overfitting.


But what is the difference between bagging and Random Forest? If we use all models as Decision trees in bagging so is it same as Random forest?

Bagging and Random Forest differ in their approach and flexibility. In bagging, any model such as Decision Trees, K-Nearest Neighbors (KNN), or Support Vector Machines (SVM) can be used as the base algorithm, while in a Random Forest, only Decision Trees are utilized.
Now what if in bagging we keep all models as Decision tree, is it same as random forest? 
No, They aren't the same due to the way they handle bootstrapping, which involves row and feature sampling, In bagging, if we select three Decision Trees and have five columns in total, we might select two columns for each tree. For instance, if we choose columns 1 and 3 for the first tree, all leaves of this fully grown Decision Tree will use only these two columns, maintaining the initial column selection for each tree and same happens for other trees too. This is tree-level column sampling.
But in decision tree column sampling occurs at each node during the formation of the tree. At every node, a subset of features is randomly selected, adding a layer of randomness at the node level, which enhances the overall randomness and robustness of the model. So node level column sampling happens here.


What are the hyperparameters?
1. max features: Number of columns to be used
2. Bootstraped: Randomly or not
3. max_samples: number of samples each tree will receive (usually 50%-75% gives best results)


# Adaboost

In Adaboost we use weak learners. Weal learners are the algorithms which gives bad accuracy or accuracy just above 50% so in adaboost we add many weak learners.

In other models we have binary (0 and 1) classes for output but here in adaboost we have -1 and +1 instead. But why so? We will see

basically here in adaboost we randomly create decision stumps (Decision stumps are the decision trees with max depth as 1) and after that we check the mis classified points and let the next model know to focus on these misclassified points more using upsampling (it means increase the value of those which are misclassified) so that the next models (weak learners) handles the misclassified point well. It's akin to making a mistake in life and advising your siblings to avoid the same path.
after that for each model, we generate an alpha value, which indicates the model's influence during prediction. If a model makes accurate predictions, it receives a high alpha value; if it performs poorly, it gets a low alpha value. This process is repeated for multiple base models, continuously improving the overall accuracy of the ensemble.

1. Initially, in AdaBoost, we assign a weight to every row, which is (1/n) (where n is the number of rows), meaning each row has the same weight at the start.

2. In the first stage, we train a decision tree with a maximum depth of 1, known as a decision stump. We generate many decision stumps but select the one that results in the maximum decrease in entropy. This becomes our model m1. We then test m1 and calculate alpha1, which depends on the error rate.

But how do we calculate it? suppose we got 3 models a, b and c and we got error rate as 100%, 0% and 50% so which one is more trustable? a is most predictible because we know it already that it will always make mistake no matter what so we can just make its prediction negative and take that as real output. Therefore, low error rates correspond to high alpha values, and high error rates correspond to negative alpha values. For a 50% error rate, alpha is 0.5. so such function is 1/2*ln(1-error/error) where error is sum of all misclassified points weights (one we initiated using 1/n)
Next, we perform upsampling by increasing the weights of the misclassified points, so the next model can focus on these points. The weight updates are as follows:
for misclassified points: new_wt= curr_wt*e^alpha1
for correctly classified points: new_wt= curr_wt*e^-alpha1
After updating the weights, we ensure their sum is 1 by normalizing them. This process is repeated, with each new model building on the mistakes of the previous ones, iteratively improving the overall performance.

For upsampling in AdaBoost, we use the new weights to create a range that starts from 0. For example, suppose our new weights are 0.166, 0.25, 0.25, 0.167, and 0.167. We create ranges as follows: 0-0.166, 0.166-0.416, 0.416-0.666, 0.666-0.833, and 0.833-1. We then generate n (the number of rows, say 5) random numbers between 0 and 1. Suppose the random numbers are 0.13, 0.43, 0.62, 0.50, and 0.8. We select the rows corresponding to these random points within the generated ranges, resulting in the selection of rows 1, 3, 3, 3, and 4, creating a new dataset.

The benefit of this approach is that rows with larger weights, which are typically the misclassified rows, have a higher probability of being selected multiple times. This ensures that the new dataset emphasizes the rows that were previously misclassified, allowing the next decision stump to focus on these difficult cases but how? 

We then train a new decision stump on this upsampled dataset, calculate a new alpha alpha2, and update the weights again. This process repeats, with each iteration focusing more on the misclassified points from the previous iterations, thus progressively improving the model's accuracy.

Looking at hyperparameters
1. base estimator: we use it to select models as this algo can be run in Decision tree, linear regression,SVM but not KNN because no feature for sample weight available

2. n_estimator: number of decision trees(here we can select anu number of DT as it stopes early if correct decision tree is found if not it goes on)

3. learning rate: generally it is 1 byt we reduce it as it usually helps to deal with overfitting as it reduces the amplitude of sample weights at each step
