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

Research Paper link
https://jmlr.org/papers/volume13/biau12a/biau12a.pdf
