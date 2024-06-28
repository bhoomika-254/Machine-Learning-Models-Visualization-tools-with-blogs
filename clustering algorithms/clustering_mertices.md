# Commonly used metrices in Clustering algorithms

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