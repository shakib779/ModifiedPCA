import numpy as np
import pandas as pd
import warnings
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def conv(num):
	num = (int) (num*1000)
	num /= 10.0
	return num


def solve(X, eig, feature, dim, beg):     # Projection Onto the New Feature Space
	end = beg + dim - 1;
	matrix_w = np.hstack((eig[beg][1].reshape(feature,1)))
	for i in range(beg + 1, end + 1):
		matrix_w = np.hstack((matrix_w.reshape(feature, i - beg), eig[i][1].reshape(feature,1)))
	Y = X.dot(matrix_w)
	return Y;


def getAccuracy(X, y, repeat):
	kf = StratifiedShuffleSplit(n_splits = repeat, test_size = 0.1, random_state = 0)
	sum = 0.0
	for train_index, test_index in kf.split(X, y):
  		X_train, X_test = X[train_index], X[test_index]
  		y_train, y_test = y[train_index], y[test_index]
  		model = svm.SVC(gamma = 'auto')
  		model.fit(X_train, y_train)
  		score = model.score(X_test, y_test)
  		sum += score
	score = sum/(repeat*1.0);
	return score


def solveLDA(X, y, dimension, selected_dimension):
	lda = LDA(n_components = selected_dimension)
	lda.fit(X, y)
	X_lda = lda.transform(X)
	return X_lda


def solvePCA(X, dimension, selected_dimension):
	mean_vec = np.mean(X, axis=0)
	cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
	cov_mat = np.cov(X.T)

	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	for ev in eig_vecs:
		np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	eig_pairs.sort()
	eig_pairs.reverse()

	X_pca = solve(X, eig_pairs, dimension, selected_dimension, 0)

	return X_pca



def solveModifiedPCA(X, y, dimension, selected_dimension):
	mean_vec = np.mean(X, axis=0)
	cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
	cov_mat = np.cov(X.T)

	eig_vals, eig_vecs = np.linalg.eig(cov_mat)

	for ev in eig_vecs:
		np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	eig_pairs.sort()
	eig_pairs.reverse()

	hi = 0.0
	pos = 0
	for i in range(dimension - selected_dimension + 1):
		X_new = solve(X, eig_pairs, dimension, selected_dimension, i)
		score = getAccuracy(X_new, y, 1)
		if(score > hi):
			hi = score
			pos = i
	
	X_mod_pca = solve(X, eig_pairs, dimension, selected_dimension, pos)

	return X_mod_pca



# *************************************************** Dataset  *******************************************************

warnings.filterwarnings("ignore")


data1 = pd.read_csv('data1.txt')
data2 = pd.read_csv('data2.txt')
data3 = pd.read_csv('data3.txt')
data4 = pd.read_csv('data4.txt')

X1 = np.array(data1.drop(['Caesarian'], 1))
y1 = np.array(data1['Caesarian'])


X2 = np.array(data2.drop(['class'], 1))
y2 = np.array(data2['class'])


data3.drop(['PatientId'], 1, inplace = True)
X3 = np.array(data3.drop(['class'], 1))
y3 = np.array(data3['class'])


data4.drop(['id', 'date'], 1, inplace = True)
X4 = np.array(data4.drop(['Occupancy'], 1))
y4 = np.array(data4['Occupancy'])

attribute = []
instance = []

attribute.append(len(X1[0]))
attribute.append(len(X2[0]))
attribute.append(len(X3[0]))
attribute.append(len(X4[0]))

instance.append(len(X1))
instance.append(len(X2))
instance.append(len(X3))
instance.append(len(X4))

#********************************    Normalization of data as different variables in data set may be having different units of measurement ******************************# 

X_std1 = StandardScaler().fit_transform(X1)
X_std2 = StandardScaler().fit_transform(X2)
X_std3 = StandardScaler().fit_transform(X3)
X_std4 = StandardScaler().fit_transform(X4)


###############################################################################
###############################################################################
selected_dimension = 2                                              ###########
repeat_kf = 1                                                       ###########
###############################################################################
###############################################################################



# *******************************************  Reduce dimension from dataset *******************************************************************#


X_pca1 = solvePCA(X_std1, attribute[0], selected_dimension)
X_lda1 = solveLDA(X_std1, y1, attribute[0], selected_dimension)
X_mod_pca1 = solveModifiedPCA(X_std1, y1, attribute[0], selected_dimension)


X_pca2 = solvePCA(X_std2, attribute[1], selected_dimension)
X_lda2 = solveLDA(X_std2, y2, attribute[1], selected_dimension)
X_mod_pca2 = solveModifiedPCA(X_std2, y2, attribute[1], selected_dimension)


X_pca3 = solvePCA(X_std3, attribute[2], selected_dimension)
X_lda3 = solveLDA(X_std3, y3, attribute[2], selected_dimension)
X_mod_pca3 = solveModifiedPCA(X_std3, y3, attribute[2], selected_dimension)


X_pca4 = solvePCA(X_std4, attribute[3], selected_dimension)
X_lda4 = solveLDA(X_std4, y4, attribute[3], selected_dimension)
X_mod_pca4 = solveModifiedPCA(X_std4, y4, attribute[3], selected_dimension)


# ************************************************** EVALUATION *********************************************************** #
accuracy = []
pca_accuracy = []
lda_accuracy = []
pca_mod_accuracy = []

accuracy.append(getAccuracy(X_std1, y1, repeat_kf))
pca_accuracy.append(getAccuracy(X_pca1, y1, repeat_kf))
lda_accuracy.append(getAccuracy(X_lda1, y1, repeat_kf))
pca_mod_accuracy.append(getAccuracy(X_mod_pca1, y1, repeat_kf))


accuracy.append(getAccuracy(X_std2, y2, repeat_kf))
pca_accuracy.append(getAccuracy(X_pca2, y2, repeat_kf))
lda_accuracy.append(getAccuracy(X_lda2, y2, repeat_kf))
pca_mod_accuracy.append(getAccuracy(X_mod_pca2, y2, repeat_kf))


accuracy.append(getAccuracy(X_std3, y3, repeat_kf))
pca_accuracy.append(getAccuracy(X_pca3, y3, repeat_kf))
lda_accuracy.append(getAccuracy(X_lda3, y3, repeat_kf))
pca_mod_accuracy.append(getAccuracy(X_mod_pca3, y3, repeat_kf))


accuracy.append(getAccuracy(X_std4, y4, repeat_kf))
pca_accuracy.append(getAccuracy(X_pca4, y4, repeat_kf))
lda_accuracy.append(getAccuracy(X_lda4, y4, repeat_kf))
pca_mod_accuracy.append(getAccuracy(X_mod_pca4, y4, repeat_kf))


# ************************************************************ EXPERIMENTAL RESULT ********************************************************************************* #


print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

print("        Classification Accuracy Table for some datasets by Projecting Data onto 2 Dimensions: ")
print(" ")
print("        +---------+--------+---------------------+---------------------+-----------------------+---------------------+")
print("        |         |        |    Classification   |    Classification   |    Classification     |    Classification   |")
print("        |Attribute|Instance|       Accuracy      |       Accuracy      |       Accuracy        |       Accuracy      |")
print("        |         |        | (Without Reduction) |    (Applying PCA)   |(Applying Modified PCA)|   (Applying LDA)    |")
print("        +---------+--------+---------------------+---------------------+-----------------------+---------------------+")

for i in range(len(accuracy)):
	print('        |%6s   | %6s | %11s%%        | %11s%%        | %11s%%          | %11s%%        |' %(attribute[i], instance[i], conv(accuracy[i]), conv(pca_accuracy[i]), conv(pca_mod_accuracy[i]),  conv(lda_accuracy[i])))
	print("        +---------+--------+---------------------+---------------------+-----------------------+---------------------+")

print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")


# ************************************************************ END ********************************************************************************* #
