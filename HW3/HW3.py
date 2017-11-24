# numeric python and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.model_selection import *

# Scoring for classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Classifiers from scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

########################################
#    Evaluate different parameters     #
########################################

svm_c_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
dt_k_vals = [2,5,10,20]

svm_list = []
dt_ig_list = []
dt_gini_list = []

svm_scores = []
dt_ig_scores = []
dt_gini_scores = []

# instantiate classifiers with each value of their parameter
for c in svm_c_vals:
	svm_list.append(LinearSVC(C=c))

for k in dt_k_vals:
	dt_ig_list.append(DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=k))

for k in dt_k_vals:
	dt_gini_list.append(DecisionTreeClassifier(criterion='gini', max_leaf_nodes=k))


D = np.genfromtxt('spambase.data', delimiter=",")
X = D[:, :-1]
y = D[:, -1]

# cross validate
for clf in svm_list:
	results = cross_validate(clf, X, y, scoring='f1_macro', cv=10, n_jobs=8, return_train_score=False)
	avg = np.sum(results['test_score'])/len(results['test_score'])
	svm_scores.append(avg)

for clf in dt_ig_list:
	results = cross_validate(clf, X, y, scoring='f1_macro', cv=10, n_jobs=8, return_train_score=False)
	avg = np.sum(results['test_score'])/len(results['test_score'])
	dt_ig_scores.append(avg)

for clf in dt_gini_list:
	results = cross_validate(clf, X, y, scoring='f1_macro', cv=10, n_jobs=8, return_train_score=False)
	avg = np.sum(results['test_score'])/len(results['test_score'])
	dt_gini_scores.append(avg)

# get c value that yielded highest F-measure for SVM
best_c_index = np.argmax(svm_scores)
best_c = svm_c_vals[best_c_index]

# get k value that yielded highest F-measure for DT-IG
best_k_ig_index = np.argmax(dt_ig_scores)
best_k_ig = dt_k_vals[best_k_ig_index]

# get k value that yielded highest F-measure for DT-Gini
best_k_gini_index = np.argmax(dt_gini_scores)
best_k_gini = dt_k_vals[best_k_gini_index]


########################################
#    Evaluate different classifiers    #
########################################


# instantiate classifiers with parameters from from parts a) and b)
best_svm = LinearSVC(C=best_c)
best_dt_ig = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=best_k_ig)
best_dt_gini = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=best_k_gini)

# instantiate LDA and RandomForest
lda = LinearDiscriminantAnalysis()
rf = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=9)

# train each classifier
best_svm.fit(X_train, y_train)
best_dt_ig.fit(X_train, y_train)
best_dt_gini.fit(X_train, y_train)
lda.fit(X_train, y_train)
rf.fit(X_train, y_train)

# record predictions from each classifier
y_pred_svm = best_svm.predict(X_test)
y_pred_dt_ig = best_dt_ig.predict(X_test)
y_pred_dt_gini = best_dt_gini.predict(X_test)
y_pred_lda = lda.predict(X_test)
y_pred_rf = rf.predict(X_test)

# calculate precisions
precision_svm = average_precision_score(y_test, y_pred_svm)
precision_dt_ig = average_precision_score(y_test, y_pred_dt_ig)
precision_dt_gini = average_precision_score(y_test, y_pred_dt_gini)
precision_lda = average_precision_score(y_test, y_pred_lda)
precision_rf = average_precision_score(y_test, y_pred_rf)
precision_scores = [precision_svm, precision_dt_ig, precision_dt_gini, precision_lda, precision_rf]

# calculate recalls
recall_svm = recall_score(y_test, y_pred_svm)
recall_dt_ig = recall_score(y_test, y_pred_dt_ig)
recall_dt_gini = recall_score(y_test, y_pred_dt_gini)
recall_lda = recall_score(y_test, y_pred_lda)
recall_rf = recall_score(y_test, y_pred_rf)
recall_scores = [recall_svm, recall_dt_ig, recall_dt_gini, recall_lda, recall_rf]

# calculate F-measures
f1_svm = f1_score(y_test, y_pred_svm)
f1_dt_ig = f1_score(y_test, y_pred_dt_ig)
f1_dt_gini = f1_score(y_test, y_pred_dt_gini)
f1_lda = f1_score(y_test, y_pred_lda)
f1_rf = f1_score(y_test, y_pred_rf)
f1_scores = [f1_svm, f1_dt_ig, f1_dt_gini, f1_lda, f1_rf]

print("best C for SVM: ", best_c)
print("best k for DT-IG: ", best_k_ig)
print("best k for DT-Gini: ", best_k_gini)
print()

print('Precision scores')
print('\tSVM: ', precision_svm)
print('\tDT-ig: ', precision_dt_ig)
print('\tDT-gini: ', precision_dt_gini)
print('\tLDA: ', precision_lda)
print('\tRF: ', precision_rf)

print('Recall scores')
print('\tSVM: ', recall_svm)
print('\tDT-ig: ', recall_dt_ig)
print('\tDT-gini: ', recall_dt_gini)
print('\tLDA: ', recall_lda)
print('\tRF: ', recall_rf)

print('F-Measures')
print('\tSVM: ', f1_svm)
print('\tDT-ig: ', f1_dt_ig)
print('\tDT-gini: ', f1_dt_gini)
print('\tLDA: ', f1_lda)
print('\tRF: ', f1_rf)

########################################
#             Plot results             #
########################################

figure, ((ax_svm, ax_dt_ig, ax_dt_gini), (ax_precision, ax_recall, ax_fmeasure)) = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))
figure.canvas.set_window_title("HW 3")
clf_labels = ('SVM', 'DT-IG', 'DT-Gini', 'LDA', 'RF')
clf_colors = ('dodgerblue', 'limegreen', 'lime', 'turquoise', 'green')
y_lim = (0.5, 1.0)

# parts a) and b)
ax_svm.set_title('SVM')
ax_svm.axes.set_ylabel('F1-Score')
ax_svm.axes.set_ylim(y_lim)
ax_svm.axes.set_xlabel('C')
ax_svm.axes.set_xticks(range(0, 5))
ax_svm.axes.set_xticklabels(svm_c_vals)
ax_svm.bar(range(0,5), svm_scores, align='center')

ax_dt_ig.set_title('DT-IG')
ax_dt_ig.axes.set_ylabel('F1-Score')
ax_dt_ig.axes.set_ylim(y_lim)
ax_dt_ig.axes.set_xlabel('Max. leaf-nodes')
ax_dt_ig.axes.set_xticks(range(0, 4))
ax_dt_ig.axes.set_xticklabels(dt_k_vals)
ax_dt_ig.bar(range(0,4), dt_ig_scores, align='center')

ax_dt_gini.set_title('DT-Gini')
ax_dt_gini.axes.set_ylabel('F1-Score')
ax_dt_gini.axes.set_ylim(y_lim)
ax_dt_gini.axes.set_xlabel('Max. leaf-nodes')
ax_dt_gini.axes.set_xticks(range(0, 4))
ax_dt_gini.axes.set_xticklabels(dt_k_vals)
ax_dt_gini.bar(range(0,4), dt_gini_scores, align='center')

# part c)
ax_precision.set_title('Avg. Class Precision')
ax_precision.axes.set_ylabel('Precision')
ax_precision.axes.set_ylim(y_lim)
ax_precision.axes.set_xticks(range(0,5))
ax_precision.axes.set_xticklabels(clf_labels)
ax_precision.bar(range(0,5), precision_scores, align='center', color=clf_colors)

ax_recall.set_title('Avg. Class Recall')
ax_recall.axes.set_ylabel('Recall')
ax_recall.axes.set_ylim(y_lim)
ax_recall.axes.set_xticks(range(0,5))
ax_recall.axes.set_xticklabels(clf_labels)
ax_recall.bar(range(0,5), recall_scores, align='center', color=clf_colors)

ax_fmeasure.set_title('Avg. Class F-Measure')
ax_fmeasure.set_ylabel('F1-Score')
ax_fmeasure.axes.set_ylim(y_lim)
ax_fmeasure.axes.set_xticks(range(0,5))
ax_fmeasure.axes.set_xticklabels(clf_labels)
ax_fmeasure.bar(range(0,5), f1_scores, align='center', color=clf_colors)
plt.subplots_adjust(wspace=0.5, hspace=0.7)

plt.show()



















































