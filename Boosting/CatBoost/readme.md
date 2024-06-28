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

