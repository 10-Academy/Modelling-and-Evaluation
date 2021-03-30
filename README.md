# Modelling Evaluation

What is Modelling and Evaluation as it relates to Machine Learning
We can explain Machine Learning Models without discussing Statistical Modelling
From Wikipedia, statistical model is a mathematical model that embodies a set of statistical assumptions concerning the generation of sample data (and similar data from a larger population). A statistical model represents, often in considerably idealized form, the data-generating process.

Statistical modelling is a method of mathematically approximating the world. Statistical models contain variables that can be used to explain relationships between other variables. We use hypothesis testing, confidence intervals etc to make inferences and validate our hypothesis. Machine learning is all about results, it is like working in a company where your worth is characterized solely by your performance. Whereas, statistical modeling is more about finding relationships between variables and the significance of those relationships, whilst also catering for prediction.

## Importance of Understanding Modelling and Evaluation
1. Recognition and Classification of Objects: Training a Machine Learning Models helps to label/annotate data and make right decisions. For an example, if you want your algorithm to recognize these two different species of animals — say a cat and dog, you need labeled images containing these two class of animals.
When your algorithm learns what are the features are important in distinguishing between two classes. It helps them to recognize and classify the similar objects in future, thus training data is very important for such classification. And if it is not accurate it will badly affect the model results, that can become the major reason behind the failure of AI project.

2. Validating the Machine Learning Model: Developing a ML model is not enough, you need to validate the model to check its accuracy/RMSE, so that you can ensure the prediction quality in real-life. To evaluate such ML model, you need another set of training data which can be also called the validation data, use to check the accuracy level/RMSE of the model in different scenario and to ensure the model generalize well.

## Key Types of Machine Learning Models (Algorithms)
1. Supervised Learning
How it works: This algorithm consist of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, [Decision Tree](https://www.analyticsvidhya.com/blog/2015/01/decision-tree-simplified/), [Random Forest](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/), [KNN](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761), [Logistic Regression](https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python) etc.

2. Unsupervised Learning
How it works: In this algorithm, we do not have any target or outcome variable to predict / estimate. It is used for clustering population in different groups, which is widely used for segmenting customers in different groups for specific intervention. Examples of Unsupervised Learning: [Apriori algorithm](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html), [K-means](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1).

3. Reinforcement Learning:
How it works: Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: [Markov Decision Process](https://www.analyticsvidhya.com/blog/2020/11/reinforcement-learning-markov-decision-process/)
 
## List of Common Machine Learning Algorithms

1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. SVM
5. Naive Bayes
6. kNN
7. K-Means
8. Random Forest
9. Dimensionality Reduction Algorithms
10. Gradient Boosting algorithms

The [article attached here](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/) explains all you need to know about all the listed algorithm, how, when and why we use them. 

## Evaluating Machine Learning Models (classification)
True positives (TP): Predicted positive and are actually positive. <br>
False positives (FP): Predicted positive and are actually negative.<br>
True negatives (TN): Predicted negative and are actually negative.<br>
False negatives (FN): Predicted negative and are actually positive<br>

1. Confusion matrix: is just a matrix of the above evaluation metrics
![confusion_matrix](https://user-images.githubusercontent.com/40719064/112985360-fbf12800-9157-11eb-9cde-4bec6dd194b8.png)

2. Accuracy: the most commonly used metric to judge a model and is actually not a clear indicator of the performance. The worse happens when classes are imbalanced.
![accuracy score](https://miro.medium.com/max/230/1*PfGgbFFjLjGYkp_lHXFvgg.png)

3. Precision: Percentage of positive instances out of the total predicted positive instances. Here denominator is the model prediction done as positive from the whole given dataset. Take it as to find out ‘how much the model is right when it says it is right’. ![precision score](https://miro.medium.com/max/105/1*LWDZT9hRYc7BAzpeZUOZrg.png)

4. Recall/Sensitivity/True Positive Rate: Percentage of positive instances out of the total actual positive instances. Therefore denominator (TP + FN) here is the actual number of positive instances present in the dataset. Take it as to find out ‘how much extra right ones, the model missed when it showed the right ones’. ![recall score](https://miro.medium.com/max/111/1*U_CKVn3iy9WN6ckfZ9_LeA.png)

5. F1-Score: it is the harmonic mean of both the Precision and Recall. One drawback is that both precision and recall are given equal importance due to which according to our application we may need one higher than the other and F1 score may not be the exact metric for it. Therefore either weighted-F1 score or seeing the PR or ROC curve can help. ![f1](https://miro.medium.com/max/533/1*rxeJQS0ALoR3pFNFjgTD6g.png)
6. ROC curve: ROC stands for receiver operating characteristic and the graph is plotted against TPR and FPR for various threshold values. As TPR increases FPR also increases.

## Evaluating Machine Learning Models (Regression)
