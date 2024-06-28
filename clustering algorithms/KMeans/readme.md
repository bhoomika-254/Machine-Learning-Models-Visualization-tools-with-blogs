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
