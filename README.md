# Machine Learning Models Visualization tools with some blogs

Till now I have uploaded
1. Gradient descent visualization tool (For GD,SGD and simulated_annealing for both convex and non convex curve)
2. Bagging Classifier visualization tool
3. Bagging Regressor visualization tool
4. Voting Classifier visualization tool
5. Voting Regressor visualization tool
6. Decision Tree visualization tool

Upcoming:
1. Random forest visualization tool
2. SVM Blog
3. XGBoost Blog
4. XGBoost visualization tool
5. Decision Tree blog
6. Notebook for all ML Models
7. Mathematical approach for all of the blogs
8. Papers
9. Notebook for visualizing few models
10. Stacking and blending Blog and notebook
11. Bagging and voting blog
12. DBSCAN Clustering Blog


# Gradient descent 
Visualization tool:

Open
```bash
gd_sgd_app.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/gd_sgd_app.py
```
and if the above dosen't work out then run 
```bash
streamlit run (gd_sgd_app.py path over your local pc)
```
Demo:

https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/67fc9ace-4351-4e2b-b1ec-3fc28482ad9c

......Blog Coming Soon........

# Decision Tree
```bash
Programmatically, decision trees are essentially a large structure of nested if-else conditions. Mathematically, decision trees consist of hyperplanes that run parallel to the axes, dividing the coordinate system into hyper cuboids.
Looking at it through an image we see
```

**Mathematically**

![WhatsApp Image 2024-06-26 at 16 52 32_9bcbd8ad](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/f5d28044-ea1d-4279-a554-5f9250ff1ac9)


![WhatsApp Image 2024-06-26 at 16 53 23_527a356d](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/8f35989c-9fe7-4788-8bd9-0118854b34ec)


**Geometrically**


![WhatsApp Image 2024-06-26 at 16 56 14_99a66417](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7032d9a1-c0d4-4be3-b441-6e083f7b6cab)

While it is easy to write these nested if and else condition just by looking at the dataset but it is tough to do these in some large dataset and for that decision Tree algorithm works.

A practical implementation of a decision tree can be seen in the game Akinator.
https://en.akinator.com/

**Few terminologies before starting off**

1. **Root Node:** The starting point of a decision tree where the entire dataset begins to split based on different features or conditions.
2. **Decision Nodes:** Intermediate nodes resulting from the splitting of the root node. They represent further decisions or conditions within the tree.
3. **Leaf Nodes:** Terminal nodes where no further splitting occurs, indicating the final classification or outcome. These nodes are also known as terminal nodes.
4. **Sub-Tree:** A portion of the decision tree, similar to a sub-graph in a graph, representing a specific section of the overall tree.
5. **Pruning:** The process of removing certain nodes in a decision tree to prevent overfitting and simplify the model.
6. **Branch / Sub-Tree:** A specific path of decisions and outcomes within the decision tree, representing a subsection of the entire tree.
7. **Parent and Child Node:** In a decision tree, a node that splits into sub-nodes is called a parent node, while the resulting sub-nodes are called child nodes. The parent node represents a decision or condition, and the child nodes represent the potential outcomes or further decisions based on that condition.

![terminilogies](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/07af1952-f7c4-4e3d-954a-3d1f48bf7d69)


**So lets look at a pseudo code for splitting column/row wise**
1. Begin with your training dataset which should have some feature variable for say classification and regression output
2. Determine the best feature in the dataset to split the data on. But now you might think how much to split? where to stop? To answer this question, we need to know about few more concepts like entropy, information gain, and Gini index which we will look below
3. Split the data into subsets that contains the correst values for the best feature this splitting basically defines a node on a tree. i.e. each node is a splitting point based on a certain feature from our data
4. Recursively generate new tree nodes by using the subset of data created in step 3

**Before diving into everything else, let's understand what entropy is**
Entropy, in simple terms, measures the uncertainty or disorder in a dataset. It can also be seen as a measure of purity and impurity.
Imagine you and your friends are deciding which amusement park to visit on Sunday. There are two choices: "Wonderla" and "Imagicaa." Everyone votes, and "Wonderla" gets 4 votes while "Imagicaa" gets 5 votes. It's hard to decide because the votes are nearly equal.
This situation illustrates disorder. With almost the same number of votes for both parks, it's tough to choose which one to visit. If "Wonderla" had 8 votes and "Imagicaa" had only 2, it would be easy to decide to visit "Wonderla" because it has the majority of votes.

Mathematical Formula: 


![entropy](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7207023d-75fb-4bbe-b94a-674fdce1e779)


Where 'Pi' is simply the frequentist probability of an elements/class 'i' in our data

If our data has class levels Yes and No we get entropy as,


![entropy 2 class](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0c593c3d-488f-4e5c-980d-afaecf64c890) 

For more than 2 class we get,


![3class](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/4273b703-c14f-4408-a1ed-1424d3ad4287)

And same goes on for n number of classes

Now lets look at it through an example


![example_cleanup](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a03217f4-cfd7-4f63-b2c3-90f31c772234)

![example 2](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/b862938f-f1f3-44f3-b878-40c2b3d3c874)

Now based upon above 2 examples we can plot a graph


![pde](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a1512bf3-9a06-4816-a6c3-b7f0926465c8)


In this graph we can see If all the probability is NO then Entropy is 0 means our system has so much of knowledge to predict that every point is NO and if all the probability is YES then also entropy is 0 means our system has so much of knowledge to predict that every point is YES.

We can get entropy easily for Categorical columns but what about continuous variable? Let us look at it through an example again


![num](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0cadb7f5-d41e-4f9c-aa87-ce697083a341)

We will see this through KDE curve 


![num graph](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/d6456dc8-4ce4-49a3-96a3-2a4cea4862ba)


Let's say the black curve is for Dataset 1 and the light blue curve is for Dataset 2
Here we can see Dataset 1 has more peak but less variance than Dataset 2 so D2 will have more entropy than D1 because a low peak and more variance => More entropy

So in a nutshell,
In a decision tree, the output is usually a simple "yes" or "no". Entropy measures the impurity of a node, indicating the degree of randomness in the data. A pure split means you should get either all "yes" or all "no" answers. For example, if a feature has 8 "yes" and 4 "no" initially, and after the first split, the left node gets 5 "yes" and 2 "no" while the right node gets 3 "yes" and 2 "no", the split is not pure. This is because both nodes still contain a mix of "yes" and "no" answers. To build a decision tree, we calculate the impurity of each split. When the impurity is 0%, making the split pure, we designate it as a leaf node. We use the entropy formula to check the impurity of different features.

1. More uncertainty means more entropy
2. For 2 class problems (Yes and No) the min-entropy is 0 and the max entropy is 1
3. For more than 2 class problems (Yes, No, and Maybe) min entropy is 1 but maximum entropy can be greater than 1.
4. For calculating entropy we can use both log base e and log base 2
5. For 3 class problem we have E= -P(Yes)log2(P(yes))-P(No)log2(P(No))-P(Maybe)log2(P(Maybe))
6. The higher the Entropy, the lower will be the purity and the higher will be the impurity


The objective of machine learning is to reduce uncertainty or impurity in the dataset. Using entropy, we assess the impurity of a specific node, but we don't know if the entropy of the parent node has decreased.

To address this, we introduce a new metric called "Information gain," which quantifies how much the parent node's entropy decreases after splitting it with a particular feature.

Information gain determines the reduction in uncertainty associated with a feature, playing a crucial role in selecting attributes for decision nodes or the root node.

Information gain=E(Parent)- (Weighted average)*E(Children) 
E(Parent)= Entropy of complete dataset

Let us understand it better through an example where we take a dataset that tells us whether one will play tennis or not


![=data](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/fa9c3067-6199-4881-b084-63c90aa295bd)


Step 1: Calculate E(Parent)


![step1](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/207b2de2-b13c-45cf-b09e-2548bdfedfb3)

Step 2: On the basics of the outlook column we will divide the dataset so for all values of sunny we will have dataset 1 then for Overcast we will have dataset 2 then for Rainy we will have dataset 3

Step 3: We calculate the weighted entropy of children  where we multiply each entropy with the weighted average.


![step2_cleanup](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/30fbb65f-1956-463a-89a7-be4ced60b415)

By the way we can see P(Overcast) entropy is 0 so when we get 0, we do no more splitting as that will be our leaf node.

Step 4 is calculating the information gain:


![step3](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/1079dd6d-b5f9-4038-9d35-513ad348d6ad)

So the information gain(or decrease in entropy/impurity) when you split this data on the basics of outlook condition column is 0.28

Step 5: Calculate Information gain for all the columns like this and whichever column has the highest information gain the algorithm will select that column to split the data

And then decision tree applies a recursive greedy search algorithm in top buttom fashion to find info gain at every level of the tree and once the leaf node is reached (entropy 0) no more splitting is done

In a nutshell, Information gain is just a Metric which is used to train decision tree. The Information gain is based on the decrease in the entropy after a dataset is split on an attribute. Constructing a decision tree is about finding attributes that return the highest Information gain.


Now just like Entropy we have another metric Gini Impurity
Gini impurity measures a dataset's impurity or disorder in decision tree algorithms. For binary classification, it evaluates the chance of incorrect classification when a data point is randomly assigned a class label based on class distribution in a node. The Gini impurity ranges from 0 (perfectly pure node with all instances in the same class) to 0.5 (maximum impurity with an equal distribution of classes). It helps identify the best feature splits to create more homogeneous data subsets, leading to accurate and reliable predictive models.


![gini](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/d773bf64-b082-463f-9a69-0e5a7ba1bbbc)

Let us again look at gini impurity through an example,


![gini example](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0d1518c8-b3ed-429f-802d-17d48cb7c511)

In nutshell we know Less gini means more information and other than that everything is just same Information Gain= Parent Gini- Weighted Gini child and just like entropy splitting criteria is same

When examining the Gini impurity curve, we observe that, unlike entropy, the maximum Gini impurity value is 0.5. This indicates a 50/50 probability, meaning there is an equal likelihood of randomly picking an instance from either class.

![gini curve](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/e7f9ec03-f058-489a-aef2-74b5b91c8e7e)

To expand on this, let's consider a binary classification problem where we have two classes, A and B. If a dataset is perfectly mixed, with half of the instances belonging to class A and the other half to class B, the Gini impurity would reach its maximum value of 0.5. This reflects maximum disorder or uncertainty because each class is equally probable.

In contrast, if the dataset were perfectly pure, with all instances belonging to a single class, the Gini impurity would be 0, indicating no uncertainty. Thus, the Gini impurity ranges from 0 (complete purity) to 0.5 (maximum impurity), helping us measure and understand the impurity levels in our dataset.

**Now what in numerical data? Instead of catagorical columns we have numerical column with n unique values?**
**Step 1** Sort the whole data on the basics of numerical column
**Step 2** Split the whole data on the basics of every value of the column

![WhatsApp Image 2024-06-26 at 18 39 37_a46bae1d](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0884d057-ab01-498c-bfb9-191426478309)

Where D is the dataset, f is the column user rating and v1,v2,v3,...vn are values of various rows of user rating

**Step 3** We Find out entropy and weighted entropy of each child

![WhatsApp Image 2024-06-26 at 18 40 00_33ad4417](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/f8b71df2-fa95-409d-a9c4-fb8953c74da2)

**Step 4** Now we try to find Information Gain of each child and we get the max{IG1,IG2.....IGn} means we get the maximum Information gain. Suppose got IG3 as the maximum so we will split on the basics of f>v3

**When to Stop Splitting?**

You might be wondering when to stop growing a decision tree. In real-world datasets with many features, the tree can become very large due to numerous splits. Such large trees take a long time to build and can lead to overfitting. This means the tree will perform well on the training data but poorly on the test data.

To address this, we can use hyperparameter tuning. One method is to set the maximum depth of the decision tree using the `max_depth` parameter. The larger the `max_depth` value, the more complex the tree. While increasing `max_depth` decreases training error, it can lead to poor accuracy on test data due to overfitting. To find the optimal `max_depth` that prevents both overfitting and underfitting, you can use GridSearchCV.

Another method is to set the minimum number of samples required for a split using the `min_samples_split` parameter. For example, if you set a minimum of 10 samples, any node with fewer than 10 samples will not be split further and will be made a leaf node. This helps control the growth of the tree and prevents it from becoming too complex.

**Pruning**

Pruning is a technique used to prevent overfitting in decision trees. It improves the tree's performance by removing nodes or branches that are not significant or have very low importance.

There are two main types of pruning:

1. **Pre-pruning**: This involves stopping the growth of the tree early. During the tree-building process, we can cut a node if it has low importance.
   
2. **Post-pruning**: After the tree has been fully built, we start pruning nodes based on their significance. This means we remove nodes that don't add much value to the model.

**Hyperparameters** 
Now let us look at few hyperparameters in sklearn class
1. **min_samples_leaf**: This represents the minimum number of samples required to be in a leaf node. Increasing this number can help prevent overfitting.(high value of mon samples leaf will give underfitting and lower value will give overfitting so we need some optimum value which we find through algos like GridsearchCV)
2. **max_features**: This determines the number of features to consider when looking for the best split.(mostly used in higher dimensions to reduce overfitting)
3. **max_depth**: It defines the depth of the tree and this is one of the major reasons in overfitting (max_depth=none we see overfitting max_depth=1 gives underfitting)

**Visualization tool:**

Open
```bash
decision_tree.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/decision_tree.py
```
and if the above dosen't work out then run 
```bash
streamlit run (decision_tree.py path over your local pc)
```
Demo:


https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/3518b8dc-ae7b-456e-b75a-bf2c3eb4b43a

......Blog Coming Soon........

# SVM

......Blog Coming Soon........

# Clustering Algorithms

# K Means

K-means is an unsupervised algorithm that also works in higher dimensions.

The first step involves deciding on the number of clusters K which is pretty weird as the algorithm doesn't determine the optimal number of clusters on its own.

Suppose we decide on 3 clusters. The model will randomly select 3 points from the dataset to serve as the initial centroids.

Next, we assign each data point to the nearest centroid based on the Euclidean distance. We calculate the distance from each data point to all centroids and assign the point to the cluster with the nearest centroid.

Once the clusters are assigned, we compute the new centroids of each cluster, typically by calculating the mean of all the points in the cluster (e.g., the mean of all IQ and CGPA values).

This process is repeated iteratively: reassigning clusters based on the nearest centroids and recalculating the centroids. The algorithm stops when the centroids no longer change from one iteration to the next, indicating convergence.

To decide the number of clusters initially in K-means, we can use the elbow method. In this method, we plot a graph between the number of clusters and the within-cluster sum of squared distances (WCSS) or inertia. Here's how it works:

1. **Start with a Single Cluster:** Calculate the WCSS for one cluster by finding its centroid and calculating the squared distance between all points and this centroid.

2. **Calculate for Multiple Clusters:** Increase the number of clusters to 2, calculate the WCSS for each cluster, and sum them up to get the total WCSS. Repeat this process for 3 clusters, 4 clusters, and so on, up to a reasonable number (e.g., 20 clusters).

3. **Plot the Elbow Curve:** Plot the number of clusters on the x-axis and the total WCSS on the y-axis. As the number of clusters increases, the total WCSS should decrease because clusters become more specific.

4. **Identify the Elbow Point:** Look for the point on the graph where the decrease in WCSS starts to level off, forming an "elbow" shape. This point indicates where adding more clusters doesn't significantly improve the clustering performance, suggesting an optimal number of clusters.

The elbow point is where the WCSS starts to stabilize, indicating the most appropriate number of clusters for the dataset. This method helps balance the trade-off between having too few or too many clusters.


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


# Hierarchical Clustering 

Types of hirerchial clustering
1. Agglomerative Hierarchical Clustering: In this approach, we start by considering each data point as an individual cluster. Then, we iteratively merge the two closest clusters into a single cluster until all points are grouped into one cluster. This method is bottom-up, and we keep a record of the merging process using dentogram.

2. Divisive clustering: Divisive clustering is the opposite of agglomerative clustering. We begin with all data points in a single cluster and iteratively split the cluster into smaller clusters until each point is an individual cluster. This method is top-down.

Unlike K-means, which relies on centroids and may struggle with datasets shaped like concentric circles, hierarchical clustering methods can effectively handle such shapes. Agglomerative hierarchical clustering, in particular, starts with each point as its own cluster and merges them based on proximity, making it well-suited for complex cluster shapes

In hierarchical clustering, when we merge the two nearest clusters, the resulting structure is stored in a dendrogram, which visually represents the hierarchy of clusters. The dendrogram shows how clusters are progressively combined into larger clusters as we move from the bottom to the top of the diagram. To obtain a specific number of clusters from the dendrogram, we can cut it at different heights or distances. At the bottom of the dendrogram, each point starts as its own cluster. As we move upwards, clusters are successively merged until, at the top of the dendrogram, all points belong to a single cluster.

Algorithm for Agglomerative Hierarchical Clustering:

1. **Initialize Proximity Matrix:** Create an n*n proximity matrix where n is the number of data points. Populate this matrix with the distances between each pair of points. If the distance metric is Euclidean distance, for example, the matrix will contain the pairwise Euclidean distances between all points.

2. **Initialize Clusters:** Treat each data point as an individual cluster. Therefore, initially, we have \( n \) clusters, each containing one point.

3. **Iteratively Merge Clusters:**
   - **Find Closest Clusters:** Identify the two closest clusters based on the proximity matrix.
   - **Merge Closest Clusters:** Combine these two closest clusters into a single cluster.
   - **Update Proximity Matrix:** Update the proximity matrix to reflect the new distances between the merged cluster and all remaining clusters. Depending on the linkage criterion. Now what linkage criterion we use? Look at the below 4 points
   
4. **Repeat Until One Cluster is Left:** Continue merging the closest clusters and updating the proximity matrix iteratively until only one cluster remains. At this point, the dendrogram or hierarchy of clusters is complete.

In sklearn we got 4 ways or rather 4 types of Agglomerative Hierarchical Clustering for inter cluster similarity(linkage criteria)
1. min(single link): we got 2 cluster and we find the distance between all points from one cluster to another and we slect the one with lowest distance (works when we have gap between our cluster as during outlier it dosent works)

2. max(complete link): we got 2 cluster and we find the distance between all points from one cluster to another and we slect the one with highest distance (maintains outliers well) disadv is if one is small other is large cluster then large cluster braeks

3. average: balance between above 2 as again we find distance between all points in clusters like above and we take the average between all clusters

4. ward: suppose we got 2 clusters then we find the centroid of all points 
after that we try to find out the distance of all points from the centroid
so distance between cluster a and b will be sq of all distances minus the variance of points inside clusters (we find mean inside clusters and calculate the distance between points and clusters) and here we reduce the variance. In sklearn by default it is ward.

but what is the ideal num of clusters? we plot dentogram and cut the longest horizontal line who is not being cut by anyone else (inter cluster similarity)


Benefits? Widely applicable in diff datasets 
And disadvantage? We cant use in large dataset like suppose we got a data set with 10^6 number of points (rows) so for proximity matrix we need 10^12 bytes space which is like 10gb so we need 10gb RAM to compute.


# DBSCAN Clustering

......Blog Coming Soon........


# Ensemble learning Methods

# Voting Classifier
Visualization tool:

Open
```bash
voting_classifier.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/voting_classifier.py
```
and if the above dosen't work out then run 
```bash
streamlit run (voting_classifier.py path over your local pc)
```
Demo:

https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/95192a43-0836-4eeb-8979-a02341fbd8f5

......Blog Coming Soon........

# Voting Regressor
Visualization tool:

Open
```bash
voting_regressor.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/voting_regressor.py
```
and if the above dosen't work out then run 
```bash
streamlit run (voting_regressor.py path over your local pc)
```
Demo:

https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/00534a61-ece5-46a2-a8b0-4df2cdb9e4e9

......Blog Coming Soon........

# Bagging Regressor
Visualization tool:

Open
```bash
bagging_regressor.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/bagging_regressor.py
```
and if the above dosen't work out then run 
```bash
streamlit run (bagging_regressor.py path over your local pc)
```
Demo:

https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/dd434c9b-2c73-41ae-ae80-48bcd3da532a

......Blog Coming Soon........

# Bagging Classifier
Visualization tool:

Open
```bash
bagging_classifier.py
```
and paste this over your terminal
```bash
streamlit run /workspaces/SGD-Visualiser/bagging_classifier.py
```
and if the above dosen't work out then run 
```bash
streamlit run (bagging_classifier.py path over your local pc)
```
Demo:

https://github.com/suvraadeep/SGD-Visualiser/assets/154406386/84abc359-cd81-4436-8215-eaa4c2334568

......Blog Coming Soon........

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


# Gradient boosting

Gradient Boosting is an ensemble learning algorithm that combines multiple weak models to create a strong predictive model.

but firstly what is additive modelling?
Machine learning is just a func of y=f(a,b,c...) where Y is output column and a,b,c are input columns and we try to find out a mathematical relationship between input and output columns
so additive modelling is F(x)=f(x1)+f(x2)........means we break into simpler model to get a larger model as whole

1. In the regression problem of gradient boosting, the first model is just a leaf node with the mean of all output column values. In classification, we use log loss. We begin by creating a new column for the predictions from the first model, which contains the mean.

Similar to other boosting methods, we need to inform the next model about the mistakes made by model 1. We use a loss function called pseudo residuals for this task, calculated as the difference between the actual output and the predicted output. For simplicity, we'll call this residuals column "res1." We create a new column for res1 corresponding to each row.

For the next model, the input columns remain the same, but the output column will now be res1. We train the new model using these inputs and outputs, and repeat this process iteratively to refine our predictions.
And the same task we do for next model (model 2) and we get predicted values as y_pred=(pred by model 1)+(learning rate*pred by model 2)
like that we get res 2= (actual- predicted) = (actual -((pred by model 1)+(learning rate*pred by model 2)))
In ideal case residual should be 0 so when we add more models then residual will tend to 0 for every model.

for 3rd model we do the same task and y_pred will be (actual -(pred by model 1)+(learning rate*pred by model 2)+(learning rate*pred by model 3))
and like this we continue



2. For classification tasks the algorithm remains the same but instead of pseudo residuals we use log loss

So in Gradient Boosting Classifier the first model is just a leaf node with the log odds (log(number of 1s/number of 0s)) as output column and we convert this log loss to Probability using (1/(1+e^(-log odds)))

It is being observed that max leaf nodes 8-32 gives best result 

........ongoing...........



# Stacking and Blending

......Blog Coming Soon........


For any required changes for any mistake I made feel free to point me out over suvradeep@iitg.ac.in
