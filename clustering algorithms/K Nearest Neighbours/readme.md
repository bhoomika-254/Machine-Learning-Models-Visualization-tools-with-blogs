# KNN

K-nearest neighbors (KNN) is a simple yet powerful algorithm based on the idea that "you are the average of the 5 people you spend the most time with." To classify a new query point, we start by deciding on a value of K, which represents the number of nearest neighbors to consider. For example, if K=3, we look at the 3 nearest neighbors.

Given a query point (x,y), we calculate the distance between this point and all other points in the dataset using a distance metric such as Euclidean distance. We then sort these distances in ascending order and select the top 3 closest points. Suppose the closest points have labels 1, 1, and 0; the majority label is 1, so we classify the query point as 1.

KNN works in any number of dimensions, making it versatile for various datasets. However, it is crucial to ensure proper scaling of the data, as KNN relies on distance calculations. Without scaling, features with larger numerical ranges can disproportionately influence the distance measurements, leading to suboptimal results.


Selecting the optimal value of K in K-nearest neighbors (KNN) is crucial for the algorithm's performance. While there is no definitive answer as it depends on the dataset, two common approaches are the heuristic method and experimentation.

1. In the heuristic approach, a commonly used rule of thumb is to set K to the sqrt(n), where n is the number of observations in the dataset. For example, if there are 100 observations, k would be approximately sqrt{100} = 10. Additionally, it is advisable to choose an odd value for K to avoid ties in binary classification, where an even K could result in an equal number of neighbors for each class (2 Yes and 2 No).

2. The experimental method involves cross-validation. In this approach, various values of K are tested, and the value that provides the highest accuracy or best performance on the validation set is selected. This method systematically evaluates different K values and identifies the optimal one based on empirical results.

By combining these approaches, one can effectively determine the best K for a specific dataset, balancing simplicity and empirical performance.

**Decision surface:**  A decision surface is used in classification scenarios as a tool to understand the performance of a K-nearest neighbors (KNN) model. To plot a decision surface, we first plot the training data and identify the maximum values on the x and y axes. Then, we generate a numpy meshgrid(we fill the whole graph with points) and then we send these generated points to KNN model and if knn model tells blue then blue region and if knn model tells red then red region for all points and we show all these points as pixels in the graph. This can be done using the mlxtend library.

**Overfitting and Underfitting in KNN:** In KNN, the value of k plays a crucial role in model performance. A low k value can lead to overfitting, where the model captures noise and fluctuations in the training data, resulting in poor generalization to new data. Conversely, a high k value can lead to underfitting, where the model is too simplistic and fails to capture the underlying patterns in the data. Therefore, it is important to find an intermediate value of k that balances the bias-variance trade-off, providing good generalization performance.

Limitation?
1. **Large Datasets:** KNN is a lazy algorithm, meaning it does no work during the training phase and performs all computations during prediction. As a result, KNN can become very slow for large datasets because it needs to calculate distances, sort them, and find the majority count for each prediction.
2. **High Dimensional Data:** KNN suffers from the curse of dimensionality, where the distance metrics become less reliable as the number of dimensions increases. This is because, in high-dimensional spaces, all points tend to be equidistant from each other, making it difficult to identify meaningful nearest neighbors.
3. **Sensitivity to Outliers:** KNN is not robust to outliers. Outliers can disproportionately affect the distance calculations and lead to incorrect classifications.
4. **Non-homogeneous Scale:** KNN is sensitive to the scale of the features. If features are on different scales, the distance calculations will be dominated by the features with larger numerical ranges. Therefore, it is essential to perform scaling (e.g., standardization or normalization) before applying KNN.
5. **Imbalanced Data:** KNN does not perform well with imbalanced datasets. If one class is much more frequent than the others, the algorithm will be biased towards the majority class, leading to poor performance on the minority class.
6. **Inference vs. Prediction:** KNN is good for inference but not for prediction in the sense that it does not provide insight into the relationship between input features and the output. It is a black-box model that does not reveal the mathematical relationship between \( x \) and \( y \), making it difficult to interpret the model's decisions.

