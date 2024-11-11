# Machine Learning Models Visualization tools with some blogs

So, I started this repo thinking, "Hey, I'll just toss up some Streamlit apps to visualize my ensemble learning models." But then I had this thought why stop there lol? Why not add some blogs and notebooks too? Because it'll totally be useful. This repo's basically my personal guidebook now lol. And hey, if anyone else wants to peek at my stuff, they can check out my blogs, notebooks, and those cool Streamlit apps I whipped up.

***Till now I have uploaded***
1. Gradient descent visualization tool (For GD,SGD and simulated_annealing for both convex and non convex curve)
2. Bagging Classifier visualization tool
3. Bagging Regressor visualization tool
4. Voting Classifier visualization tool
5. Voting Regressor visualization tool
6. Decision Tree visualization tool
7. XGBoost Blog
8. Decision Tree blog
9. Notebook for visualizing decision tree each step
10. DBSCAN Clustering Blog

Upcoming:
1. Random forest visualization tool
2. SVM Blog
3. Notebook for all ML Models
4. Mathematical approach for all of the blogs
5. Stacking and blending Blog and notebook
6. Bagging and voting blog and notebook
7. Gradient descent blog


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

**Programmatically, decision trees are essentially a large structure of nested if-else conditions. Mathematically, decision trees consist of hyperplanes that run parallel to the axes, dividing the coordinate system into hyper cuboids.**
Looking at it through an image we see

**Mathematically**

![WhatsApp Image 2024-06-26 at 16 52 32_9bcbd8ad](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/f5d28044-ea1d-4279-a554-5f9250ff1ac9)


![WhatsApp Image 2024-06-26 at 16 53 23_527a356d](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/8f35989c-9fe7-4788-8bd9-0118854b34ec)


**Geometrically**


![WhatsApp Image 2024-06-26 at 16 56 14_99a66417](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7032d9a1-c0d4-4be3-b441-6e083f7b6cab)

While it is easy to write these nested if and else condition just by looking at the dataset but it is tough to do these in some large dataset and for that decision Tree algorithm works.

A practical implementation of a decision tree can be seen in the game Akinator.
https://en.akinator.com/

I have made this notebook to visualise the formation of decision tree step-wise using this amazing library dtreeviz which you can checkout through

https://www.kaggle.com/code/suvroo/dtreeviz-a-decision-tree-visualization-tool

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
3. Split the data into subsets that contains the correct values for the best feature this splitting basically defines a node on a tree. i.e. each node is a splitting point based on a certain feature from our data
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

Before we start with everything let us look at some commonly used metrics for clustering algorithms

**```1. Silhouette score```**
The Silhouette score is a metric used to evaluate the quality of clustering in data science. It measures how similar an object is to its own cluster compared to other clusters.

The Silhouette score combines two factors:
**Cohesion (a):** How close an object is to other objects in its own cluster.
**Separation (b):** How far an object is from objects in the nearest neighboring cluster.
The Silhouette score for a single data point 𝑖
i is calculated as:

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/64a92115-ebb4-4eff-845e-32fbf09548de)

**a(i)** is the average distance from the point 𝑖 to all other points in the same cluster.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/26eafa2d-51bd-4150-b0b5-bd640d924bfa)

**b(i)** is the average distance from the point 𝑖 to all points in the nearest cluster that is not the one it belongs to.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0ed3aeef-e0a3-45a5-a41a-8ddcbc3f7737)

**The Silhouette score ranges from ```-1 to +1```**
**+1:** Indicates that the data point is well clustered. It is far from neighboring clusters and close to the points in its own cluster.
**0:** Indicates that the data point is on or very close to the decision boundary between two neighboring clusters.
**-1:** Indicates that the data point might have been assigned to the wrong cluster, as it is closer to a neighboring cluster than to its own cluster.

In some cases we will get more than 2 Points so how do we check the quality of each cluster? We can examine the ```Silhuette plot```
We should ideally get 0 negative points but in some cases it is comparable

```Let's understand through an example:```

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/24b3e542-3846-429f-ac87-09afe2ae4f3b)

```In this example we can see that for k=5 we are getting some negative points for 4th cluster (0 based indexing) ```

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a47d0bfd-8001-4e3f-bb01-e9df68ea0a1b)

```Also in this example we can see that for k=6 we are getting some negative points for 3rd cluster (0 based indexing) ```

Comparing both we can say that K=5 is more suited for our model because it is giving us less neagtive values


**In a nutshell** the overall Silhouette score for a clustering solution is the average Silhouette score of all data points. This provides a measure of how appropriately the data has been clustered, with a higher average score indicating better-defined clusters.

**```2. Elbow Curve```**

The elbow curve plots the cost (also known as the inertia or within-cluster sum of squares) against the number of clusters. The cost represents how compact the clusters are. The goal is to identify the point where adding more clusters does not significantly improve the model, indicating the optimal number of clusters.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/9b132bd0-4906-4875-9a56-b4d45a4eefcc)

Where
**X-axis:** Number of clusters (k).
**Y-axis:** Cost (within-cluster sum of squares).

**Steps to Create an Elbow Curve**
**1.** Run the clustering algorithm (usually in k-means) for different values of k (say from 1 to 10 and can go up to n where n represents number of instances).
**2.** Calculate the cost for each k. In the case of k-means, this cost is the sum of squared distances from each point to its assigned cluster center (centroid).
**3.** Plot the cost against the number of clusters (k).

After plotting we try to find the Elbow Point which is the optimal number of clusters where the cost reduction diminishes significantly. This point represents a trade-off between the number of clusters and the improvement in clustering performance. Before Elbow Point adding more clusters significantly reduces the cost and after Elbow Point Adding more clusters yields diminishing returns, meaning it does not significantly improve the clustering quality.

**So in a nutshell** The elbow curve is a simple yet effective method for selecting the number of clusters in clustering algorithms. By plotting the cost against the number of clusters, it helps identify the point where adding more clusters does not lead to a significant improvement, thus determining the optimal number of clusters for the dataset.

**```3. Dendrogram```**
A dendrogram is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering algorithms. It is a crucial tool for visualizing the arrangement of the clusters produced by hierarchical clustering which we will look below but just to be concise there are two types of hierarchical clustering algorithm
**Agglomerative approach:** Start with each data point as a single cluster and merge the closest pairs of clusters iteratively until all points are in one cluster.
**Divisive approach:** Start with all data points in a single cluster and recursively split them into smaller clusters.

Example of Using a Dendrogram using sets:
Consider a dataset with points A,B,C,D:

**1.** Start: Each point is its own cluster: {A},{B},{C},{D}.
**2.** Merge closest clusters: If A and B are closest, merge them: {AB},{C},{D}.
**3.** Continue merging: If C is next closest to AB, merge: {𝐴𝐵𝐶},{𝐷}
**4.** Final merge: Merge D with 𝐴𝐵𝐶 ,we get {ABCD}.
A dendrogram would depict these merges, with the vertical position of each merge indicating the distance at which the merge occurred.

And diagramatically we get dendrogram like,

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/4ea5afc4-2cd1-4c48-a7fa-2dcfb1b1592a)

here, The length of the branches represents the distance or dissimilarity between clusters. Shorter branches mean smaller distances (more similar clusters), and longer branches indicate larger distances (less similar clusters).

Now how do we decide the number of cluster from this?
We just look for the longest vertical line which is not being cut by any other horizontal line and the number of leafs will be the desire number of clusters like for example

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/3ae18c25-ab30-4895-998c-30d32e161474)

Here in this case 2 is the number of clusters.

**So in a nutshell**, A dendrogram represents the nested grouping of clusters and illustrates the order in which clusters are merged or split. The leaves of the dendrogram represent individual data points, while the branches show how clusters are formed.

# K Means

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/fc27fe35-6b14-4600-bab9-4997236ba063)


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

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0d0792b2-3328-448d-b982-baefbcb9eef1)

But in some case elbow curve might be confusing so such as:

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/8d4ff589-d37b-4ffc-a459-06ead193bc01)

The graph above shows the reduction of a distortion score as the number of clusters increases. However, there is no clear "elbow" visible. The underlying algorithm suggests 5 clusters. A choice of 5 or 6 clusters seems to be fair.
Another way to choose the best number of clusters is to plot the silhuette score in a function of number of clusters. Let's see the results.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/edd7d432-e371-4cc7-8e25-7a662ccb4f63)

But unfortunately Silhouette score method indicates the best options would be 5 or 6 clusters. Let's compare both. Though this wont really happen in other datasets as usually we will get 1 or 2 K only so we just we need train for both and check for the best results.


In this case we can say 8 is the number of cluster

Code: 

```bash
from sklearn.cluster import KMeans
wcss = []
X = df[['X' , 'Y']] #change X and Y based upon your desired one
for i in range(1,11):
    km = KMeans(n_clusters=i,max_iter=300,tol=0.0001,algorithm='elkan')
    km.fit_predict(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
```
So in a nutshell
There are 3 main steps in K-Means algorithm:
1. Split samples into initial groups by using seed points. The nearest samples to these seed point will create initial clusters.
2. Calculate samples distances to groups’ central points (centroids) and assign the nearest samples to their cluster.
3.The third step is to calculate newly created (updated) cluster centroids.
Then repeat steps 2 and 3 until the algorithm converges.

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

Dendrogram for a particular dataset:

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/e2ca6e6c-18a3-4e8d-a340-2c29391ab4fe)

As we can see we can cut 5 lines without any break hence 5 is the number of clusters formed

Benefits? Widely applicable in diff datasets 
And disadvantage? We cant use in large dataset like suppose we got a data set with 10^6 number of points (rows) so for proximity matrix we need 10^12 bytes space which is like 10gb so we need 10gb RAM to compute.

Research paper link 
https://arxiv.org/pdf/1109.2378


# DBSCAN Clustering

![dbscan](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/fc67ce52-51c5-4eab-8394-f33d9d5111b9)


DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**

The first question that arises is: Despite the effectiveness of KMeans on many datasets, why would we consider other algorithms like DBSCAN? What are the limitations of DBSCAN?

1. Unlike KMeans, where we manually determine the number of clusters using techniques like the elbow curve or silhouette score, DBSCAN automates this process based on density, potentially avoiding ambiguity in results from these manual techniques.
   
2. KMeans is highly susceptible to outliers, whereas DBSCAN is more robust in handling them due to its density-based approach.
   
3. KMeans operates under the assumption of clusters being spherical or globular around centroids, making it less effective with irregularly shaped clusters, which DBSCAN can handle more effectively. Like in fails in below dataset

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a9661ad0-d148-41b4-9311-8bab20f0cb9e)

As the name implies, DBSCAN operates on the principle of density-based clustering. The core idea is to identify clusters as areas of high density separated by areas of low density (which may be outliers or noise). In practical terms, DBSCAN distinguishes clusters by examining the density of points.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/9e0b2b5f-0827-4789-b9ad-10319a736baf)

In the diagram provided, the blue and yellow regions represent clusters because they contain a high concentration of points, while the red region is classified as sparse because it has fewer points. DBSCAN uses a parameter called ```"epsilon" (ε)``` to determine the neighborhood around each point and another parameter ```"minPts"``` to decide on the minimum number of points required to form a dense region (cluster). Points that do not meet these criteria are considered outliers or noise.

Therefore, DBSCAN effectively partitions the dataset based on the density of points, identifying dense regions as clusters and isolating sparse regions as noise or outliers. This approach allows DBSCAN to handle datasets with irregular shapes and varying densities more robustly compared to centroid-based algorithms like KMeans.

Now let us look more deeply what are these ```MinPts``` and ```epsilon (ε)``` through a diagram

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/3dc43305-d49b-4287-af98-7fffe83d0196)

Let us say epsilon is 1 unit and MinPts is 3 so that means in a circle or rather a cluster of unit 1 if we have more than 3 points we will consider that as a cluster so the larger circle is a cluster but the left hand side circle is treated as sparse because it got less than 3 points inside it. So in short Epsilon is radius and MinPts is threshold. And luckily these are the only two parameters we need to tune in DBSCAN. and do we tune we will look below.

Before moving forward let us look some terminologies:

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/ea4110d4-747e-47b1-aa6d-44f6ceb068f9)

In DBSCAN, points are categorized into three types based on their local neighborhood density:
- **Core points:** These are points where the number of neighboring points within a specified radius (epsilon, ε) is greater than or equal to a threshold (MinPts). Core points are central to dense regions and help define the boundaries of clusters.
  
- **Border points:** These points lie within the epsilon neighborhood of a core point but do not meet the density requirement themselves (i.e., they have fewer neighbors than MinPts). Border points are part of a cluster but are not dense enough to be considered core points or we can say it should have atleast one core point in it to be considered as border point.

- **Noise points:** Points that do not qualify as core or border points are considered noise. They do not belong to any cluster and are typically isolated points or outliers in the dataset.

In a nutshell DBSCAN identifies clusters by examining the density of points relative to each other, where core points act as the foundation of clusters, border points connect clusters to their outskirts, and noise points are isolated from any significant cluster structure.

Now let's explore another term: "Density-connected points." Two points, A and B, are considered density-connected if they can be grouped into the same cluster. This grouping occurs when there exists a path of core points between A and B, and every consecutive pair of core points along this path are within a specified distance (epsilon, ε).

Density-connectedness holds true unless two conditions fail:
1. The distance between A and B exceeds epsilon.
2. One of the points (either A or B) is not a core point but rather a border or noise point.

DBSCAN algorithm:
1. Start by initializing values for MinPts and epsilon (ε).
2. Classify each point as a core point, border point, or noise point based on its neighborhood density.
3. For each unclustered core point:
   - (a) Create a new cluster.
   - (b) Add all unclustered points that are density-connected to the current core point into this cluster.
4. Assign each unclustered border point to the nearest core point's cluster.
5. Leave noise points unassigned, as they do not belong to any cluster.

Let's illustrate the DBSCAN algorithm using a set of points: {a,b,c,d,e,f,g,h}.

1. **Initialize Parameters:**
   - Assume MinPts = 3 (minimum number of points in epsilon neighborhood to be considered core).
   - Assume epsilon (ε) = 2 (maximum distance to be considered neighbors).

2. **Identify Core, Border, and Noise Points:**
   - Calculate the epsilon neighborhood for each point to determine core, border, and noise points.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/84bc7210-ca4c-4ec8-bd33-1b7bf4cd2096)

   Based on MinPts = 3:
   - Core points: {c, d, e, f, g}
   - Border points: {a, b, h}
   - Noise points: None (since all points are either core or border).

3. **Cluster Formation:**
   - Start clustering with unvisited core points.

   **Cluster 1 (Starting with c):**
   - Add c to Cluster 1.
   - Add d and e to Cluster 1 (density-connected to c).
   - Add f to Cluster 1 (density-connected to d).
   - Cluster 1: {c, d, e, f}

   **Cluster 2 (Starting with g):**
   - Add g to Cluster 2.
   - Add h to Cluster 2 (density-connected to g).
   - Cluster 2: {g, h}

4. **Assign Border Points:**
   - Assign each border point to the nearest core point's cluster.
   - Assign a to Cluster 1 (nearest core point: c).
   - Assign b to Cluster 1 (nearest core point: c).
   - Border points: {a, b} are assigned to Cluster 1.

5. **Final Clusters and Noise:**
   - Final clusters: Cluster 1: {a, b, c, d, e, f}, Cluster 2: {g, h}
   - Noise point: None (since all points are clustered).

***The question now is how we should set the initial values for MinPts and epsilon (ε), because these parameters significantly impact the effectiveness of the algorithm.***

If we choose inappropriate values for MinPts and epsilon, it can lead to suboptimal clustering results or even failure of the algorithm to identify meaningful clusters. Therefore, selecting suitable values for these parameters is crucial for the successful application of DBSCAN.

Well there are many ways to do this but my what I actually do is I run a loop for different MinPts and epsilon (ε) and then I use a heatplot which shows shows how many clusters were generated by the DBSCAN algorithm for the respective parameters combinations to get the range in which clusters are formed

![download](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/cf42e9bf-1034-42b8-acfb-ea19b0eb7d1d)

The heatplot above shows, the number of clusters vary from 17 to 4. However, most of the combinations gives 4-7 clusters. To decide which combination to choose I will use silhuette score and I will plot it as a heatmap again.

![download](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/bfdacb18-a73e-4d4c-ac87-e69a5b2a2ad8)

And for the following silhuette score heatmap we get global maximum as 0.26 for eps=12.5 and min_samples=4. So  MinPts will be 4 and epsilon (ε) will be 12.5.

Also one more point I forgot to mention that DBSCAN labels all outliers/noise as cluster '-1'

Now you must have felt that there are many advantages of using DBSCAN like it is robust to outliers, we dont need to specify the number of clusters, we can find any arbitary shaped clusters also there are only 2 hyperparameters to tune. But we got some disadvantages too like it is very sensitive to hyperparameters, it fails in when all sparse points or in a single core points and it dosent predict.

Visualization tool for DBSCAN
https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/

Research paper Link
https://arxiv.org/pdf/1810.13105.pdf

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

Research Paper link
https://jmlr.org/papers/volume13/biau12a/biau12a.pdf

# Adaboost

In Adaboost we use weak learners. Weak learners are the algorithms which gives bad accuracy or accuracy just above 50% so in adaboost we add many weak learners.

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

Research Paper Link
https://www.researchgate.net/publication/321583409_AdaBoost_typical_Algorithm_and_its_application_research/fulltext/5a29be04a6fdccfbbf8185df/AdaBoost-typical-Algorithm-and-its-application-research.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19

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

Research Paper Link
https://www.researchgate.net/publication/259653472_Gradient_Boosting_Machines_A_Tutorial/fulltext/02d519920cf2c60a84412f57/Gradient-Boosting-Machines-A-Tutorial.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19

# Stacking and Blending

......Blog Coming Soon........

# XGBoost

XGBoost, short for eXtreme Gradient Boosting, is an ensemble learning algorithm widely used in supervised learning tasks like regression and classification. It constructs predictive models by combining multiple individual models, typically decision trees, in an iterative manner. The algorithm operates by sequentially adding weak learners to the ensemble, each subsequent learner aiming to rectify the errors made by its predecessors. During training, XGBoost employs gradient descent optimization to minimize a specified loss function.
Key features of the XGBoost algorithm include its capability to handle intricate data relationships, incorporation of regularization techniques to prevent overfitting, and utilization of parallel processing for efficient computation. These attributes make XGBoost a powerful tool for developing accurate and robust predictive models across various domains.
 So in short XGBoost, a machine learning algorithm within the gradient boosting framework, enhances upon traditional gradient boosting with significant improvements:

1. **Speed:**
   - **Parallel processing:** XGBoost divides work across multiple threads for faster sequential training.
   - **Optimized data structure:** It stores data column-wise, optimizing memory access and boosting computational efficiency.
   - **Cache awareness:** Utilizes cache memory for efficient histogram-based training.
   - **Out-of-core computing:** Handles large datasets by loading chunks sequentially into RAM.
   - **Distributed computing:** Supports distributed computing frameworks, enabling parallel processing across different models.
   - **GPU Support:** Leverages GPU acceleration for even faster computation.

2. **Flexibility:**
   - **Cross-platform:** XGBoost is compatible with various platforms.
   - **Multiple language support:** Supports integration with multiple programming languages.
   - **Integration with other libraries and tools:** Easily integrates with existing machine learning libraries and tools.
   - **Supports diverse ML problem statements:** Suitable for regression, classification, ranking, and other machine learning tasks.

3. **Performance:**
   - **Regularized learning objective:** Implements regularization techniques to prevent overfitting.
   - **Handling missing values:** Can handle missing data points seamlessly during training.
   - **Sparsity-aware split finding:** Efficiently identifies and handles sparse data.
   - **Efficient split finding and tree pruning:** Uses optimized algorithms for finding splits in decision trees and pruning unnecessary branches.

XGBoost is celebrated for its superior performance, flexibility, and speed, making it a preferred choice across various machine learning applications. This article provides an overview of XGBoost and explores its practical use-case scenarios.

We will look at its algorithm in Regression and Classification problem statement separately

***XGBoost for Regression***

 Let us take a dataset where we got input column as cgpa and output column as Package
 
 ![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/8f6a8246-8a6f-4a16-b915-355f44a78e20)

Stage 1: Just like gradient boosting we use base predictor as mean, i.e we fill the output from model 1 with the mean of output column as model 1 is nothing but mean of output column

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/1d3579c7-4973-427e-80b3-34e798605724)

Stage 2: We calculate the residual for each row and fill this in a new column res 1

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/b4a55740-5bc1-4eed-aa4b-b9a32d121300)


In XGBoost unlike Gradient boost pr Random forest we don't use Gini Impurity or entropy as splitting criteria but here we have  Similarity score given by:

![sim](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/554ecf55-21e8-4eda-84af-777a836b07e8)

Now let us look at the construction of the tree step by step
1. We first calculate the similarity score for the root node as shown below

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/9eae7b8a-825a-46b8-9db1-1db27baa860e)

2. Then we calculate the mean of 2 consecutive rows after sorting the input columns which will be our splitting criteria and we get

3. 
![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/544be1e6-ce3b-4c65-a2aa-13618c3a05c7)


Here, we identify the splitting criteria with potential threshold values of 5.85, 7.2, and 8.25. Our objective is to construct a decision tree that maximizes the gain, similar to the approach used in Random Forests. However, in XGBoost, we calculate the gain using a slightly different formula:

```{Gain} = [[{Similarity score of left node} + {Similarity score of right node}] - {Similarity score of root node}]```

So in a nutshell to determine the optimal split, we perform the following steps:

1. **Calculate Similarity Scores:**
   - Compute the similarity score for the root node using the entire dataset.
   - For each threshold value (5.85, 7.2, and 8.25), split the data into left and right nodes and calculate the similarity scores for these nodes.

2. **Compute Gain:**
   - For each threshold value, calculate the gain using the formula above. The gain represents how much the split improves the homogeneity of the resulting nodes.

3. **Select the Best Split:**
   - Compare the gains obtained from the different threshold values.
   - The threshold with the highest gain is selected as the optimal splitting point.

![WhatsApp Image 2024-06-28 at 22 25 57_8e43f10c_cleanup](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/b81f4dba-4b2b-4250-a489-a357e58df98b)

Now that we've determined our maximum gain to be 17.52, we set the root node threshold at 8.25. From here, we continue applying the same process.
For the left child node, we have values of 0.7, -2.8, and -1.3, and for the right child node, we have 3.7. We then consider new threshold values of 5.85 and 8.25, as shown in the image below (sorry for the handwriting lol).

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/e184e309-471e-4b04-aa3e-53b41726db3e)

Then we get two of these trees as

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/d9f7990a-7aa0-4e60-b974-f2b82d0104f0)

Ans just like previous step for root node we calculate the gain as 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/942c0444-27ab-4a75-ac70-1db2587d62a7)

And this is how our final decision tree looks like 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/c0a20fc3-cf61-4221-9c72-2ae2c803d434)

We could split further if desired, but doing so might lead to overfitting, so we will stop here. With our final tree for model 2 established, we now need to calculate the output using the following method:

![WhatsApp Image 2024-06-27 at 19 06 23_48bcd5f7](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/aa5e7df4-d718-4968-be50-c1e660361098)

Now for the time being we will keep lambda as 0 so get our output value for each node as

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/e5b7a6b1-75f2-4132-bcad-b05c60772382)

After constructing the tree, we need to determine the model 2 output value for each row by mapping it through the tree. The output is calculated using the formula:

```{Model 2 Output for each row} =[{Model 1 Prediction for that row} + [{Learning Rate}*{Model 2 Prediction for that row}]]```
Where model 1 prediction is nothing but mean of all output columns. Subsequently, we calculate the residuals based on this updated prediction.
And after doing all these we get our new residual as

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0b876173-81e3-46ea-bbb1-bb710c19f36d)

As we know, the residual represents the error, and our goal is to minimize this error, ideally reaching a state where the residual is 0 or close to 0. To achieve this, we repeat the entire process until the residuals are minimized. In this case, we create a new tree by treating CGPA as the input column and the residual 2 as the output column, and then continue iterating this process.

And just for some basic info we are given 2 approach for this in the original research paper which is ```exact greedy algorithm``` and another is ```approximate algorithm``` so in small dataset we use exact greedy and in large datasets we use approximate algorithm


***XGBoost for classification***

In classification problem the workflow is same its just that we use log loss as an extra function.
We will again look at a dataset for the construction of tree

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/919fd7ee-e5db-485f-8c27-761acd90fdd0)

In stage 1, we again use the base estimator as the mean. However, since the output column contains binary values, we need to convert these binary values to log(odds) values (the same approach used in logistic regression).

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0228b418-3f1e-4d4f-bfd6-163026077c1b)

and for the following case we get

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/e696d573-3509-4290-9119-d796d6a03930)

The log odds value we calculated is approximately 0.4055. This becomes our first prediction from our base estimator. However, it seems quite odd that a prediction of whether someone will be placed or not results in a value of 0.4055. This value doesn't clearly indicate "yes" or "no," but rather falls somewhere in between. This is a limitation of using log odds directly.
To make the prediction more interpretable, we need to convert these log odds back to a probability. This conversion will give us a clearer indication of the likelihood of an event occurring, allowing us to understand the prediction in terms of probability rather than log odds.
So we convert log odds to probabilty using

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7dca1019-0a68-44bd-a950-9eac8f3a1515)

And for the following dataset we took we get Probability as 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/d05921ee-8b3e-4451-a696-d62dd6f33730)

And after all these hard work we finally calculated our residual 1 and we finally our table looked like

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/01a38de3-a062-4f24-92ff-af3bfcc00d93)

Now just like the regression problem, we calculate the similarity score but here in this case the formula for the similarity score changes a bit

![sim3](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a8683948-b08c-47a0-952e-b652ac15e0ce)

so we calculate similarity score for [-0.6,0.4,-0.6,0.4,0.4] by keeping lambda as 0 we get

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/fc0e2fe5-ac4e-4853-a4c2-0ad1d3886db4)

which equals to 0 and we got our similarity score for the root node

Similarly to a regression problem of XGBoost, we compute the mean of two consecutive rows to determine the optimal splitting criteria. This approach helps us identify the maximum gain achievable from splitting the data effectively.
And we get splitting criteria as [5.97,6.67,7.62 and 8.87]
So now we plot a tree for each to get the maximum gain

![WhatsApp Image 2024-06-29 at 01 51 03_79721220_cleanup](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/ff198213-63e6-40fc-b659-090bf03d2c2d)
![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/92f6eb0b-a5e0-4a21-92fa-266b85dd95ca)

Let us assume 2.22 is the maximum gain so we get our threshold as 7.62 and that will be our splitting criteria for our root node.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/d41170af-8f6f-4e36-a9ef-3d488b062b48)

We can split further but for the time being lets not so now we need to calculate output value for each node and here also the formula changes a bit as compared to regression problem

![lol](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/2460afed-4664-4e73-beab-a1816326aaea)

So for our tree we calculate the output value for each node we get

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/80a27c05-3e6c-4ddd-ad52-9328f0c2d7ac)

And now we need to calculate the output values for each row and the formula remains the same
```{Model 2 Output for each row} =[{Model 1 Prediction for that row} + [{Learning Rate}*{Model 2 Prediction for that row}]]```
But note that this will give us output as log odds value so again we need to convert these to probability we again use 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7dca1019-0a68-44bd-a950-9eac8f3a1515)

And calculating residual 2 we finally we get our updated table for stage 2 as

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/2c4a76f7-a6b1-43a2-bd03-8d48e1947291)

As we know, the residual represents the error, and our goal is to minimize this error, ideally reaching a state where the residual is 0 or close to 0. To achieve this, we repeat the entire process until the residuals are minimized. In this case, we create a new tree by treating CGPA as the input column and the residual 2 as the output column and then continue iterating this process.

XGBoost Research Paper: 
https://browse.arxiv.org/pdf/1603.02754


# CatBoost

Let us look at first what is the need of CatBoost when we have XGBoost, LightGBM or any other ensemble models for that let us look at its benifits

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7a274bb0-34c4-4715-ab95-934ba3c89b34)

1. From the example above, we observe that the first column contains text data, the second column contains categorical data, and the third column contains numerical data. Typically, preprocessing is required for other ML Models for categorical and text data. For categorical data, techniques like One-Hot Encoding or Label Encoding are used, and for text data, methods such as TF-IDF or Word2Vec are applied. However, with CatBoost, we don't need to perform these preprocessing steps. CatBoost automatically handles these types of data. It can be used without any explicit preprocessing to convert categories into numbers, as it transforms categorical values into numerical representations using various statistics on combinations of categorical features and combinations of categorical and numerical features.

2. It minimizes the need for extensive hyper-parameter tuning and reduces the risk of overfitting, resulting in more generalized models. Despite this, CatBoost still offers a variety of parameters that can be adjusted, such as the number of trees, learning rate, regularization, tree depth, fold size, bagging temperature, among other
   
3. Typically, machine learning models perform better with increasing amounts of data. However, CatBoost is an exception, as it delivers strong performance even with minimal data. This unique capability makes CatBoost particularly useful in scenarios where large datasets are unavailable, ensuring effective model training and accurate predictions with limited information.

Till now how did we approach for columns with categories using OHE?

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/b4781b02-6a74-4e1d-9b48-56459898e215)

One-Hot Encoding can lead to several issues, particularly when dealing with high cardinality categorical features. It creates a new binary feature for each unique category, which can result in a very high-dimensional and sparse feature space. This increases memory usage and computational complexity, and may also lead to overfitting.

For this, we use Target-Based Encoding.
![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/2eba00ee-9a10-426c-8feb-7bed76030676)

Suppose we want to encode the category 'BNG'. We calculate the average of all target values for 'BNG', which means ((10 + 10)/2), resulting in 'BNG' being encoded to 10. We apply the same process for 'Java' and 'Python'. In the end, our data will look something like this.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/c89f9495-d105-43bb-882d-9bb31adc3a95)

However, there is a significant issue with this approach: it can lead to data leakage. By using the target values directly to encode the categorical features, we inadvertently introduce information from the target variable into the feature space. This can artificially inflate the model's performance during training but result in poor generalization to new data. This leakage is undesirable as it undermines the model's ability to learn from the data independently.

But in CatBoost we used a special type of encoding which is being derived from Target-Based Encoding known as Ordered Target Encoding which addresses these issues by transforming categorical values into numerical values based on the target variable. This method captures the relationship between the categorical feature and the target, reducing dimensionality and sparsity. It also helps in improving model performance by providing more meaningful numerical representations of the categorical features.

We use this formula to encode 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/0959140f-2d3f-41e6-a92f-56f71a2f6f6d)

Here, the prior is a parameter, typically set to 0.05 in most cases. 
- **current_count**: This represents the total number of instances in the training dataset with the current categorical feature. Essentially, it counts how many times we have encountered this category with the same target before the current row.
- **max_count**: This counts how many times we have seen this category before the current row, regardless of the target value.\
  
Now let us look at it through an example

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/2bf9b69b-11d9-44de-9a6c-5bdc1a26a24c)

Here in 4th row (0 based indexing) BNG will be encoded to 0.8333333333 because current_count will be 2 as we got 2 BNG with target value as 1 above 4th row and max_count will be 2 as we got 2 rows above with value as BNG. and same we go for other rows too.

In summary, CatBoost avoids data leakage when encoding categorical variables by processing each row as if it were the only row in the dataset at that moment. This means that for the first row, CatBoost treats it as if it is the only data available, ignoring all subsequent rows. As it processes each row, it continuously updates the encoding based on the data seen so far, without using any information from rows that come after. This approach ensures that the encoding for each row is based solely on the preceding data, thereby preventing any information from future rows from leaking into the model.

Now let us look at how it constructs tree and we will look at it through an example (Below all image credits goes to StatQuest)

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/18c3d86c-9da4-47bc-b229-106143760643)

Each time CatBoost creates a tree, it begins by randomizing the rows of the training dataset. After shuffling the data, it then applies ordered target encoding to the categorical features. If a categorical feature has only two categories, they are replaced with 0s and 1s.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a007aa28-0ff1-4203-87c0-516d0ddd7233)


**Step 1:** Unlike Gradient Boosting and XGBoost, where the initial prediction is set to the mean of the target column, CatBoost initializes the first model prediction to 0 for each row.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/617a9a2c-12f4-4a14-a409-543682ab5753)


**Step 2:** Similar to Gradient Boosting and XGBoost, CatBoost then calculates the residuals. The residual is determined by subtracting the predicted value from the observed value, following the same formula: residual = observed - predicted.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/39732023-6208-475a-8a52-ce6448c4924a)


**Step 3:** From this step is where the magic of catboost relies because here we construct a tree where we sort the table based upon the encoded values

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/bd460b0f-3ab5-46fc-92ba-e98a1478c38f)

**Step 4:** Now same as XGBoost we calculate the similarity score which is nothing but the average of two consecutive row values 
For Regression problem similarity score is given by


![sim](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/92520370-5ae3-4b98-a309-d96f3133cee4)


For Classification problem similarity score is given by


![sim2](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/c00d4425-d172-4900-ad8a-64c0a251686e)

So after calculating similarity score for the above dataset we get,

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/ff71f0d6-945e-4466-abfe-1f62f82fbe22)

So 0.4 and 0.29 are the potential similarity score and now we construct tree

**Step 5:** in CatBoost involves initializing all leaf outputs to 0. Subsequently, each row's residual (current output) is assigned to a specific leaf.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/7ce1df8a-fbc3-481b-bdb9-a341df992814)

For instance, the first row's residual of 1.81 is placed in the right leaf because the 'Favorite Color' value of 0.05 exceeds the threshold of 0.04. The current output value for this row remains 0.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/02cbb39d-5b45-43c6-ae3f-c4f07f35bcd0)

Moving to the second row, with a residual of 1.56 placed in the right leaf, the leaf's output value is then updated to the average of the two residuals in the leaf, resulting in a value of 1.6. This iterative process continues, updating leaf outputs based on assigned residuals, thereby refining predictions progressively as more rows are processed down the tree.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/5eb1a8e8-ac94-41f3-9971-b5746b85da3c)

And same we do for other row values too as mentioned below step by step

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/aa7817af-a103-4139-9880-b4cda4c8eccb)

And after all these updates we get our table as 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/a03782a6-564d-40a7-a89a-ce8c6a1a58f8)

Now, how do we assess the quality of predictions for each threshold? CatBoost quantifies this by computing the cosine similarity (Yes the same in Word2Vec) between the column of leaf outputs and the residuals. This metric helps gauge how effectively each threshold aligns the model's predictions with the actual data, guiding CatBoost in selecting thresholds that optimize model performance and accuracy.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/8a900eec-6797-469c-8450-8df9d24586d5)
 
Where A is the residual column and B is leaf output column
And for this table we get cosine similarity as 


![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/5738246c-1d38-45fa-9eb4-8fa59791c484)


Which equals to 0.71

Next, we repeat the same process for the threshold of 0.29, iterating through the tree updates as illustrated below.

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/14b9e6ff-9d1e-4de6-9fe7-7f993d620755)

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/f23a5533-abb9-477f-bf29-37e72566f016)

Also we get the final table as 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/f3bf83aa-968c-46b6-8fb5-48f39a5b8f77)


After that we calculate the cosine similarity for this table and we get

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/5deae99c-adda-4a88-a399-6e1f8065fe57)

Which equals to 0.79
 
Upon evaluating the first tree with a threshold of 0.04, we find a cosine similarity of 0.71. In contrast, the second tree, utilizing a threshold of 0.29, yields a higher cosine similarity of 0.79. CatBoost, aiming for optimal performance, selects the threshold associated with the higher cosine similarity—in this case, 0.29. This approach ensures that subsequent tree expansions maintain consistency by evaluating splits based on their cosine similarity, thereby enhancing model effectiveness and predictive accuracy as the tree structure grows.

After this we need new prediction and we get the new prediction by using ```(1st prediction + (learning rate* leaf output))```  where 1st prection is 0 (the one we initiated earlier)

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/60e28f6f-2ddc-4194-8fa7-1fc28db24cde)


After updating the predictions using the new threshold (0.29), we calculate the residuals (Residual 2), which form the output column for the subsequent tree. Following this, we revert the encoded colors back to their original categorical values. The dataset is then shuffled again and re-encoded using ordinary target encoding. The table is sorted based on the encoded values of color, and the process of tree expansion continues by repeating these steps. This iterative approach ensures that each new tree in CatBoost builds upon the previous ones, refining predictions and improving model accuracy with each iteration.


When CatBoost encounters a dataset with more than one input column, it still constructs symmetric or oblivious decision trees. 

![image](https://github.com/suvraadeep/Machine-Learning-Models-Visualization-tools-with-blogs/assets/154406386/222da25b-ca3b-46bf-b91d-2cb60b62897e)

CatBoost's approach to building trees remains consistent regardless of the number of input columns in the dataset. It continues to prioritize symmetric or oblivious decision trees for several reasons:

1. **Simplicity and Weak Learners**: Symmetric decision trees use the same threshold for each node at the same level, making them simpler compared to traditional decision trees. This simplicity aligns with the principle of Gradient Boosting or other boosting algorithms, where the goal is to combine many weak learners to form a strong predictor. By intentionally using weaker learners (symmetric trees), CatBoost enhances the ensemble's predictive power through aggregation.

2. **Computational Efficiency**: Symmetric trees offer computational advantages by reducing the number of unique decisions (thresholds) to evaluate at each level. Instead of tracking different thresholds for different paths, CatBoost computes decisions uniformly across nodes at the same level, optimizing computational efficiency during prediction. This efficiency becomes particularly valuable in large-scale applications where fast inference times are crucial.

3. **Consistent Decision Making**: Using symmetric trees ensures consistent decision-making processes across different branches of the tree. Regardless of the specific path taken through the tree, all nodes at a given level ask the same questions about the input features. This consistency simplifies the decision-making process and facilitates faster prediction calculations.

In summary, CatBoost's use of symmetric or oblivious decision trees across datasets with multiple input columns underscores its commitment to leveraging simpler, weaker learners for improved ensemble performance and computational efficiency. This approach aligns with the foundational principles of Gradient Boosting, where incremental improvements through sequentially added trees lead to robust and accurate predictive models.

Research Paper Link
https://arxiv.org/pdf/1706.09516

An amazing book to understand the Elements of Statistical Learning deeply which I personally found useful
https://hastie.su.domains/Papers/ESLII.pdf


For any required changes for any mistake I made feel free to point me out over suvradeep@iitg.ac.in
