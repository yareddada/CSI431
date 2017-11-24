# CSI431
These are the first three homework assignments from CSI 431: Data Mining. I've included the original pdf assignment documents and the data files used.

# HW 1
This assignment involved dimensionality reduction with PCA. I was given a 10-dimensional data set, and was tasked with reducing it to 2 dimensions. The assignment required calculating the covariance matrix with 3 different formulas, considering the computational complexity of each, computing the principal components, then projecting the data onto the top 2.

Results were presented as two graphs, a scatterplot of the reduced data, and a line graph of the two principal components and their magnitude in each of the original 10 dimensions. I wasn't familiar with matplotlib when I submitted it, so the graphs are ugly and unlabeled.

# HW 2
This assignment involved creating a single-split decision tree classifier using three different measures of impurity: entropy, the Gini index, and the CART measure. I debugged my classifiers by testing them against sklearn's DecisionTreeClassifier. I decided to leave that in the code when I submitted it.

The CART impurity measure wasn't working properly, but I managed to hack around it and got 100%.

# HW 3
This assignment involved using sklearn's cross validation module to compare different classifiers: (linear) SVM, Decision trees with different impurity measures, LDA, and random forest.

First, I tested different values of the parameter C for SVM and max_leaf_nodes for the decision trees. I then compared the classifiers with their best parameters against LDA and random forest classifiers. Comparisons were based on average class precision, average class recall, and F1-score.

Results were presented as 6 bar graphs.
