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

So initially we assign a weight to every row which is 1/n (n is number of rows) means same wight for all initially.

After that in stage 1 we train a decision tree with max depth 1 or can say we train a decision stump,Intuitively we know we will get many decision stumps but we will select the one which got max decrease in entropy and like this we got our model m1 and we check the prediction on m1 using test column now we calculate alpha1 which depends upon error rate.
But how do we calculate it? suppose we got 3 models a, b and c and we got error rate as 100%, 0% and 50% so which one is more trustable? a, because we know it will always make mistake no matter what so we can make the prediction negative and take that as real output so in low error we need high alpha and if error is more we need some negative alpha, and on 50% error we get 0.5 alpha so such function is 1/2 ln(1-error) where error is sum of all misclassified points wt (one we initiated initially using 1/n)

next we do upsampling where we increase the wt of misclassified points so that next model gets to know what mistake previous model made like in which points it made mistake.
for misclassified points: new_wt= curr_wt*e^alpha1
for correctly classified points: new_wt= curr_wt*e^-alpha1
after that we make a new column with updated wt also make sure that the sum of updated weight is 1 so basically we normalise it.




Initially, in AdaBoost, we assign a weight to every row, which is \( \frac{1}{n} \) (where \( n \) is the number of rows), meaning each row has the same weight at the start.

In the first stage, we train a decision tree with a maximum depth of 1, known as a decision stump. We generate many decision stumps but select the one that results in the maximum decrease in entropy. This becomes our model \( m1 \). We then test \( m1 \) and calculate \( \alpha1 \), which depends on the error rate.

To determine \( \alpha1 \), consider we have three models \( a \), \( b \), and \( c \) with error rates of 100%, 0%, and 50%, respectively. Model \( a \) is actually the most predictable because it always makes mistakes, so we can simply invert its predictions. Therefore, low error rates correspond to high \( \alpha \) values, and high error rates correspond to negative \( \alpha \) values. For a 50% error rate, \( \alpha \) is 0.5. The formula used is \( \frac{1}{2} \ln \left(\frac{1-\text{error}}{\text{error}}\right) \), where error is the sum of the weights of all misclassified points (initially set to \( \frac{1}{n} \)).

Next, we perform upsampling by increasing the weights of the misclassified points, so the next model can focus on these points. The weight updates are as follows:
- For misclassified points: \( \text{new\_wt} = \text{curr\_wt} \times e^{\alpha1} \)
- For correctly classified points: \( \text{new\_wt} = \text{curr\_wt} \times e^{-\alpha1} \)

After updating the weights, we ensure their sum is 1 by normalizing them. This process is repeated, with each new model building on the mistakes of the previous ones, iteratively improving the overall performance.








now for upsampling we make a range starting from 0 like 0-0.166 then 0.166-0.416 the 0.416-0.666 and so on, and we generate n(number of rows say in this case 5) random numbers from 0-1 and and suppose we goy .13 then .43 then .62 the .50 then .8 and whereever these random points will fall in the range generated we select that row and we selected 1,3,3,3,4 and create a new dataset with that. now whats the benefit of doing this? we can see 3 is picked up more number of times so row 3 have larger range and that row is misclassified so now we train a new decision stump and generate alpha 2 and update wt and we go on repeating
