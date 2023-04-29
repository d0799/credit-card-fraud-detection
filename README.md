# Detection-Credit-fraud-detection
This is the final project for the Artificial Intelligence course, CSCI-6600 -02 (Spring-2023), by Darshan kumar Munireddy (00760186), Bala Tej Kumar Thanneru (00762025) and Lalithya Krishna Garaga (00764409).

This project involves training a data set with various machine learning techniques and assessing the effectiveness of each model to identify credit card fraud.

## Table of Contents: 
+ [Data Set](#Data_Set) </br>
+ [Machine Learning models used](#Machine_Learning_models_used) </br>
+ [How we used these models in project](#How_we_used_these_models_in_project) </br>
+ [Performance](#Performance) </br>

## <a name="Data_Set"></a> Data Set 

From April to September 2005, this dataset comprises information on default payments, demographic variables, credit data, payment history, and bill statements of credit card clients.

There are 24 features to choose from:

**Demographic Information**
- ID: ID of each client
- LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family / supplementary credit)
- SEX: Gender (1=male, 2=female)
- EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- MARRIAGE: Marital status (1=married, 2=single, 3=others)
- AGE: Age in years

**Repayment Status for past 6 months**

**(-1 = pay duly, 1 = payment delay for one month, 2 = payment delay for two months, ..., 9 = payment delay for nine months and above)**
- PAY_0: Repayment status in September, 2005 
- PAY_2: Repayment status in August, 2005 (scale same as above)
- PAY_3: Repayment status in July, 2005 (scale same as above)
- PAY_4: Repayment status in June, 2005 (scale same as above) 
- PAY_5: Repayment status in May, 2005 (scale same as above)
- PAY_6: Repayment status in April, 2005 (scale same as above)

**Amount of bill statement for past 6 months**
- BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
- BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
- BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
- BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
- BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
- BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)

**Amount of previous payment for past 6 months**
- PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
- PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
- PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
- PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
- PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
- PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

**Target Variable**
- default payment next month: Default payment (1=yes, 0=no)
***

## <a name="Machine_Learning_models_used"> </a> Machine Learning models used 
**SVM**

The support vector machine (SVM) is a supervised machine learning technique that can be used for classification and regression tasks. Although it can be used for both, it is mainly used for classification problems. In SVM, each value at a particular coordinate in the algorithm represents a point in n-dimensional space, where n is the number of features in the data. For each data point, a feature value is assigned to its corresponding point in the n-dimensional space. Next, a superplane is located that can clearly separate the two classes and the data is classified accordingly.

**How it works**

A simple linear SVM classifier works by drawing a straight line between two classes. In other words, one side of the line represents one category while the other side represents another category. There are infinite possible lines that can be drawn. The linear SVM algorithm is better than other algorithms like the nearest neighbor because it selects the optimal line for classifying the data points. The line is chosen to separate the data and to be as far away as possible from the nearest data points. To understand machine learning terminology, it's helpful to use 2D examples with multiple data points on a grid. The goal is to classify these data points correctly, so the line that connects the two closest points and separates the other data points is the desired line. The two closest data points are used as a reference vector to locate this line, and it is called the decision boundary.

**Logistic Regression**

Logistic regression, also known as logit regression, binary logit, or binary logistic regression, is a statistical analysis method used for predicting the outcome of a dependent variable based on past data. It is commonly used for binary classification problems. Logistic regression models the log of the outcome's odds ratio and the coefficients of the model are estimated using maximum likelihood estimation since there is no closed-form solution, unlike linear regression. Regression analysis is a predictive modeling approach used to determine the relationship between a dependent variable and one or more independent variables.

**How it works**

Logistic regression uses a "S" shaped logistic function instead of a straight regression line to predict binary outcomes with two maximum values, 0 or 1. The curve of the logistic function represents the probability of an event, such as whether cells are cancerous or not, or whether a mouse is obese based on its weight. Logistic regression is a significant machine learning technique because it can predict probabilities and classify new data using both continuous and discrete datasets. Logistic regression can categorize observations based on various types of data and can quickly identify the most useful factors for classification.

**Naive bayes**

The Naive Bayes classifier is a probabilistic machine learning model used for classification tasks. The classifier is based on the Bayes theorem, which is the fundamental concept underlying the model.

**How it works**

Bayes' theorem enables us to calculate the probability of event A occurring given that event B has already occurred. In the context of Naive Bayes classifier, B represents the evidence, A represents the hypothesis, and the predictors or features are assumed to be independent. This assumption implies that the presence of one attribute has no influence on the others, hence the term "naïve" is used to describe the model.

**Random Forest**

Random forest is a supervised learning algorithm that creates a "forest" by constructing an ensemble of decision trees. The decision trees are typically trained using the "bagging" method, which involves combining several learning models to improve the overall output. The random forest algorithm is known for its ability to reduce overfitting and improve the accuracy and stability of predictions.

**How it works**

Random forest is a supervised learning algorithm that generates a collection of decision trees, typically trained using the "bagging" method. The key concept behind bagging is that combining multiple learning models results in improved overall performance.
Random forest is a beneficial algorithm that can handle both classification and regression problems, which are the most common types of machine learning problems today. As classification is considered as a fundamental part of machine learning, let's delve deeper into how random forest can be used in classification. 

The hyperparameters of random forest are comparable to those of decision tree or bagging classifier, but the good news is that you can use the classifier-class of random forest instead of having to combine a decision tree with a bagging classifier. In addition, you can use the algorithm's regressor to handle regression tasks with random forest.

As the trees grow in random forest, more unpredictability is introduced to the model. When dividing a node, it examines the optimal feature from a random subset of features instead of just the most significant feature. This creates more diversity in the model, resulting in a better outcome.

In random forest, only a random subset of features is used to split a node, which can increase the randomness of the trees. Instead of searching for the optimal threshold, using random thresholds for each feature can make the trees even more random, similar to a standard decision tree.

 ## <a name="How_we_used_these_models_in_project"> </a> How we used these models in project
 
 - Performed training on data set.
- Predicting the model by using test data set.
- Evaluated the model by using F1 score,K Fold Cross validation and confusion matrix.
- Testing the trained model with new dataset

## <a name="Performance"> </a> Performance
**Following is the performance of each models used.**
![image](https://user-images.githubusercontent.com/95928967/145607620-8f56e6bf-5f14-4886-a5ff-6506d5abd7b8.png)

![image](https://user-images.githubusercontent.com/95928967/145636792-b7e5e418-02c6-4583-999e-9d0701a92992.png)
# credit-card-fraud-detection
