# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 2nd feature : its normalized losses in use as compared to other cars.

# Importing the dataset
dataset = pd.read_csv('/home/aspiring1/Documents/imports-85.data.csv', header=None, na_values='?')
dataset.columns = ['symboling', 'normalized-losses','make', 'fuel-type',
                   'aspiration','num-of-doors', 'body-style', 'drive-wheels',
                   'engine-location', 'wheel-base', 'length', 'width',
                   'height', 'curb-weight','engine-type',
                   'num-of-cylinders','engine-size','fuel-system','bore',
                   'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                   'city-mpg','highway-mpg','price']
dataset['num-of-doors'][dataset['num-of-doors'] == 'four'] = 4
dataset['num-of-doors'][dataset['num-of-doors'] == 'two'] = 2
dataset.drop('normalized-losses', axis=1,inplace = True)

dataset.head()

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [3,16,17,19,20,23]])
X[:, [3,16,17,19,20,23]] = imputer.transform(X[:, [3,16,17,19,20,23]])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])
labelencoder_X3 = LabelEncoder()
X[:, 2] = labelencoder_X3.fit_transform(X[:, 2])
labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])
labelencoder_X5 = LabelEncoder()
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])
labelencoder_X5 = LabelEncoder()
X[:, 6] = labelencoder_X5.fit_transform(X[:, 6])
labelencoder_X6 = LabelEncoder()
X[:, 12] = labelencoder_X6.fit_transform(X[:, 12])
labelencoder_X7 = LabelEncoder()
X[:, 13] = labelencoder_X7.fit_transform(X[:, 13])
labelencoder_X8 = LabelEncoder()
X[:, 15] = labelencoder_X8.fit_transform(X[:, 15])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)
#classifier.score(X_test,y_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

#from sklearn.linear_model import LogisticRegression
#logisticRegr = LogisticRegression(solver = 'lbfgs')
#logisticRegr.fit(X_train, y_train)
#logisticRegr.predict(X_test)
#logisticRegr.score(X_test,y_test)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Using SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf',C=1111,gamma=0.1, random_state = 0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# Applying k-fold cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf,  X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [ {'C' : [1, 10, 100, 1000] , 'kernel' : ['linear']},
                {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}
]
grid_search = GridSearchCV(estimator = clf, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Using Decision Tree for classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)

# Using Random Forest for classification
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
classifier = rfc()
classifier=classifier.fit(X_train, y_train)
Y_prediction2=classifier.predict(X_test)
print("Accuracy for RFC  : ",accuracy_score(y_test,Y_prediction2))

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
classifier = rfc()
classifier=classifier.fit(X_train, y_train)
Y_prediction2=classifier.predict(X_test)
print("Accuracy for RFC  : ",accuracy_score(y_test,Y_prediction2))

# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying xgboost algorithm
# Fitting Xgboost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)

# Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
