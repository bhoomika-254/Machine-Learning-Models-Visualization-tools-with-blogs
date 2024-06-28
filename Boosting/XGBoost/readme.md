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

# XGBoost Regression

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


# XGBoost Classification

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