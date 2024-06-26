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

