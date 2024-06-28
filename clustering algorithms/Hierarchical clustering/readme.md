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