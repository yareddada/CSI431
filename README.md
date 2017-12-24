# CSI431
These are a few of my homework assignments from CSI 431: Data Mining. I've included the original pdf assignment documents and the data files used.

# HW 2
This assignment involved creating a single-split decision tree classifier using three different measures of impurity: entropy, the Gini index, and the CART measure. I debugged my classifiers by testing them against sklearn's DecisionTreeClassifier. I decided to leave that in the code when I submitted it.

The CART impurity measure wasn't working properly, but I managed to hack around it and got 100%.

# HW 3
This assignment involved using sklearn's cross validation module to compare different classifiers: (linear) SVM, Decision trees(using entropy and Gini index), LDA, and random forest.

First, I used k-fold cross-validation to test different values of the parameter C for SVM, and max_leaf_nodes for the decision trees. I then compared the classifiers with their best parameters against LDA and random forest classifiers. Comparisons were based on average class precision, average class recall, and F1-score.

Results were presented as 6 bar graphs.
