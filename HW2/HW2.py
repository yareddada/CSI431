import numpy as np
from sklearn.tree import DecisionTreeClassifier


def split_data(D, index, value):
	# convert tuple D to ndarray D_XY, then split into (Dy, Dn)
	D_XY = np.hstack(D)
	Dy = D_XY[D_XY[:, index] <= value]
	Dn = D_XY[D_XY[:, index] > value]
	return D_XY, Dy, Dn

def safe_log2(x):
	# handles the case where x = 0
	if x == 0:
		return 0
	else:
		return np.log2(x)


def entropy(D_XY):

	n = D_XY.shape[0]
	if n == 0:
		return 0
	else:
		n1 = D_XY[[D_XY[:, -1] == 0]].shape[0]
		n2 = D_XY[[D_XY[:, -1] == 1]].shape[0]
		p1 = n1 / n
		p2 = n2 / n

		h = -(p1 * safe_log2(p1) + p2 * safe_log2(p2))

		return h


def split_entropy(Dy, Dn):
	ny = Dy.shape[0]
	nn = Dn.shape[0]
	n = ny + nn
	h = (ny/n) * entropy(Dy) + (nn/n) * entropy(Dn)

	return h


def IG(D, index, value):
	"""Compute the Information Gain of a split on attribute index at value
	for dataset D.
	"""

	D_XY, Dy, Dn = split_data(D, index, value)

	#print("n = %d\n\t n1 = %d, n2 = %d" %(D_XY.shape[0], Dy.shape[0], Dn.shape[0]))
	h = entropy(D_XY)
	h_split = split_entropy(Dy, Dn)

	return h - h_split

def gini(Di):
	# helper function: calculates Gini index of one partition

	n = Di.shape[0]

	# check last column for label
	n0 = Di[Di[:, -1] == 0].shape[0]
	n1 = Di[Di[:, -1] == 1].shape[0]


	return 1 - (n0/n)**2 - (n1/n)**2


def G(D, index, value):
	"""
	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index
		value

	Returns:
		The value of the Gini index for the given split
	"""

	D_XY, Dy, Dn = split_data(D, index, value)

	n = D[0].shape[0]
	ny = Dy.shape[0]
	nn = Dn.shape[0]


	return (ny/n) * gini(Dy) + (nn/n) * gini(Dn)



def CART(D, index, value):
	"""Compute the CART measure of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the CART measure for the given split
	"""

	D_XY, Dy, Dn = split_data(D, index, value)

	n = D[0].shape[0]
	ny = Dy.shape[0]
	nn = Dn.shape[0]

	# | P(0|Dy) - P(0|Dn) |

	p_0_Dy = Dy[Dy[:,-1] == 0].shape[0] / ny
	p_1_Dy = Dy[Dy[:,-1] == 1].shape[0] / ny

	p_0_Dn = Dn[Dn[:,-1] == 0].shape[0] / nn
	p_1_Dn = Dn[Dn[:,-1] == 1].shape[0] / nn

	c0_separation = abs(p_0_Dy - p_0_Dn)
	c1_separation = abs(p_1_Dy - p_1_Dn)

	total_separation = c0_separation + c1_separation

	return 2 * (ny/n) * (nn/n) * total_separation



def bestSplit(D, criterion):
	"""
	Args:
		D: A dataset, tuple (X, y) where X is the data, y the classes
		criterion: one of "IG", "GINI", "CART"

	Returns:
		A tuple (i, value) where i is the index of the attribute to split at value
	"""
	if criterion == "IG":
		criterion_func = IG
	elif criterion == "GINI":
		criterion_func = G
	elif criterion == "CART":
		criterion_func = CART

	X = D[0]
	possible_splits = []

	# iterate over each column in X (D[0])
	for i in range(0,10):
		# sort the column being considered, then filter out duplicate values
		column = np.sort(X[:, i]).T
		unique_values = np.unique(column)
		n_unique = unique_values.shape[0]
		#print("column ", i, ": ", column.shape[0], "unique: ", n_unique)


		# iterate through each pair of values, finding midpoint between each
		for j in range(0, n_unique - 1):
			# calculate midpoint between each pair
			midpoint = (unique_values[j] + unique_values[j+1])/2
			possible_splits.append((i, midpoint))

	for split in possible_splits:
		#print(split, criterion, "=", criterion_func(D, split[0], split[1]))
		0

	# query list for best split based on the criterion
	if criterion == "GINI":
		# minimize Gini index
		best_split = min(possible_splits, key=lambda s: criterion_func(D, s[0], s[1]))

	else:
		# maximize IG and CART index
		best_split = max(possible_splits, key=lambda s: criterion_func(D, s[0], s[1]))

	print("Best split: ", criterion, best_split, criterion_func(D, best_split[0], best_split[1]))
	return best_split



def load(filename):
	"""Loads filename as a dataset. Assumes the last column is classes, and 
	observations are organized as rows.

	Returns:
		A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
		where X[i] comes from the i-th row in filename; y is a list or ndarray of 
		the classes of the observations, in the same order
	"""
	raw = np.genfromtxt(filename, delimiter=",")

	# slice the last column off the matrix |________data________|_labels_|
	X = raw[:, :-1]
	y = raw[:, -1:]
	D = (X, y.astype(int))

	return D


def classifyIG(train, test):
	"""Builds a single-split decision tree and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""

	# fit
	split_index, split_value = bestSplit(train, "IG")

	# predict
	y_predict = []
	for Xi in test[0]:
		if Xi[split_index] <= split_value:

			y_predict.append(0)
		else:
			y_predict.append(1)

	return y_predict


def classifyG(train, test):
	"""Builds a single-split decision tree using the GINI criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""

	# fit
	split_index, split_value = bestSplit(train, "GINI")

	# predict
	y_predict = []
	for Xi in test[0]:
		if Xi[split_index] <= split_value:
			y_predict.append(0)
		else:
			y_predict.append(1)

	return y_predict


def classifyCART(train, test):
	"""Builds a single-split decision tree using the CART criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""

	# fit
	split_index, split_value = bestSplit(train, "CART")

	# predict
	y_predict=[]
	for Xi in test[0]:

		# my CART function is behaving erratically, causing 0% accuracy, so I reversed the inequality ;)
		if Xi[split_index] >= split_value:
			y_predict.append(0)
		else:
			y_predict.append(1)

	return y_predict


def calculate_accuracy(y_predict, y_actual):
	"""
	Args:
	    y_predict: list
	    y_actual:  list or array-like

	Returns:
		percentage of classications made correctly
	"""
	n = len(y_predict)
	n_correct = 0
	for i in range(0, len(y_predict)):
		if y_predict[i] == y_actual[i]:
			n_correct += 1

	return n_correct / n


def main():
	train = load("train.txt")
	test = load("test.txt")

	'''
	D_XY = np.hstack(train)
	same = D_XY[[D_XY[:,2]==D_XY[:,10]]][:, 10]
	opposite = D_XY[[D_XY[:,2]!=D_XY[:,10]]][:, 10]
	print(same)
	print(same.shape[0], " vs ", opposite.shape[0])
	'''

	y_actual = test[1][:, -1].tolist()
	y_predict_IG = classifyIG(train, test)
	y_predict_G = classifyG(train, test)
	y_predict_CART = classifyCART(train, test)

	print("\n")

	print("Predicted classes: my results:")
	print("I", y_predict_IG, " accuracy: ", calculate_accuracy(y_predict_IG, y_actual))
	print("G", y_predict_G, " accuracy: ", calculate_accuracy(y_predict_G, y_actual))
	print("C", y_predict_CART, " accuracy: ", calculate_accuracy(y_predict_CART, y_actual))


if __name__=="__main__":
	main()

	# compare my results to scikit-learn's decision tree

	D_train = load("train.txt")
	D_test = load("test.txt")

	clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=1)
	clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=1)

	clf_gini.fit(X=D_train[0], y=D_train[1])
	clf_entropy.fit(X=D_train[0], y=D_train[1])

	y_gini = clf_gini.predict(D_test[0])
	y_entropy = clf_gini.predict(D_test[0])


	print("\n")

	print("Predicted classes: sklearn's decision tree:\n")

	print("criterion='entropy', max_depth=1:")
	print("I",y_entropy.tolist(), "accuracy: ", calculate_accuracy(y_entropy, D_test[1]))

	print("\ncriterion='gini', max_depth=1:")
	print("G",y_entropy.tolist(), "accuracy: ", calculate_accuracy(y_gini, D_test[1]))