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
