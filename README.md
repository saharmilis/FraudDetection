# ExerciseBiocath
classify with imbalanced classes and with as little fp as possible.


# Objective

Attached a pickle file with 3 lists in the following order: [X_train, Y_train, X_test]. 

Q: Our objective is to classify, where the class 1 is the minority class and we would like to detect as many of it with as little fp. How would you tackle this question? The construction of some of the features is quit rigid, but others depend on some factors which can be reconsider or changed a bit. Which features would be most helpful to improve?



# Discussion

There is a number of ways to deal with imbalance. each technique has its pros & cons, and ob course its limit.
First I'll do simple things, display how I understand the data, and then use the "big guns".

##### Classification with boosting or bagging
Bagging and Boosting are similar in that they are both ensemble techniques, where a set of weak learners are combined to create a strong learner that obtains better performance than a single one. Also, they can be used to improve the accuracy of Classification & Regression Trees.

##### Undersampling
Select a subsample of the sets of zeros such that it's size matches the set of ones. There is an obvious loss of information, unless you use a more complex framework (for a instance, I would split the first set on 9 smaller, mutually exclusive subsets, train a model on each one of them and ensemble the models).

##### Oversampling
Produce artificial ones until the proportion is 50%/50%. My previous employer used this by default. There are many frameworks for this (I think SMOTE is the most popular, but I prefer simpler tricks like Noisy PCA).

##### One Class Learning
Just assume your data has a few real points (the ones) and lots of random noise that doesn't physically exists leaked into the dataset (anything that is not a one is noise). Use an algorithm to denoise the data instead of a classification algorithm.

##### Cost-Sensitive Training
Use a asymmetric cost function to artificially balance the training process.

