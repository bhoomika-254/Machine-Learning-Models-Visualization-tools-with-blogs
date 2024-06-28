# DBSCAN-Density based clustering algorithm

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
