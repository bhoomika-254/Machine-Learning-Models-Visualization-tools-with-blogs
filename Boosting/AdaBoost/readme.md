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

