from KNN import *
from arff import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Part 1
# DEBUGGING DATASET RESULTS
mat = Arff("datasets/seismic-bumps_train.arff",label_count=1)
mat2 = Arff("datasets/seismic-bumps_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

KNN = KNNClassifier(label_type='classification', weight_type='inverse_distance')
KNN.fit(train_data, train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data, test_labels)
np.savetxt("seismic-bump-prediction.csv", pred, delimiter=',',fmt="%i")
print("DEBUG DATASET")
print("Accuracy = [{:.4f}]".format(score))
print()

# EVALUATION DATASET RESULTS
mat = Arff("datasets/diabetes.arff",label_count=1)
mat2 = Arff("datasets/diabetes_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

KNN = KNNClassifier(label_type='classification', weight_type='inverse_distance')
KNN.fit(train_data, train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data, test_labels)
np.savetxt("diabetes-prediction.csv", pred, delimiter=',',fmt="%i")
print("EVALUATION DATASET")
print("Accuracy = [{:.4f}]".format(score))
print()


# Part 2
mat = Arff("datasets/magic_telescope_train.arff",label_count=1)
mat2 = Arff("datasets/magic_telescope_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

# WITHOUT NORMALIZTION
KNN = KNNClassifier(label_type='classification', weight_type='none')
KNN.fit(train_data, train_labels)
score = KNN.score(test_data, test_labels)
print("NON-NORMALIZED ACCURACY")
print("Accuracy = " + str(score))
print()

# WITH NORMALIZATION
x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

KNN = KNNClassifier(label_type='classification', weight_type='none')
KNN.fit(train_data_normalized, train_labels)
score = KNN.score(test_data_normalized, test_labels)
print("NORMALIZED ACCURACY")
print("Accuracy = " + str(score))
print()

# DIFFERENT K VALUES
k_vals = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []

for k in k_vals:
    print(k)
    KNN = KNNClassifier(label_type='classification', weight_type='none', k=k)
    KNN.fit(train_data_normalized, train_labels)
    accuracies.append(KNN.score(test_data_normalized, test_labels))

x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, accuracies)
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()


# Part 3
mat = Arff("datasets/housing_train.arff",label_count=1)
mat2 = Arff("datasets/housing_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

KNN = KNNClassifier(label_type='regression', weight_type='none')
KNN.fit(train_data_normalized, train_labels)
score = KNN.score(test_data_normalized, test_labels)
print("UNWEIGHTED HOUSING MSE")
print("MSE = " + str(score))
print()

# DIFFERENT K VALUES
k_vals = [1, 3, 5, 7, 9, 11, 13, 15]
mses = []

for k in k_vals:
    print(k)
    KNN = KNNClassifier(label_type='regression', weight_type='none', k=k)
    KNN.fit(train_data_normalized, train_labels)
    mses.append(KNN.score(test_data_normalized, test_labels))

x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, mses)
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()


# Part 4
# MAGIC TELESCOPE
mat = Arff("datasets/magic_telescope_train.arff", label_count=1)
mat2 = Arff("datasets/magic_telescope_test.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

# DIFFERENT K VALUES
k_vals = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []

for k in k_vals:
    print(k)
    KNN = KNNClassifier(label_type='classification', weight_type='inverse_distance', k=k)
    KNN.fit(train_data_normalized, train_labels)
    accuracies.append(KNN.score(test_data_normalized, test_labels))

x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, accuracies)
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HOUSING
mat = Arff("datasets/housing_train.arff", label_count=1)
mat2 = Arff("datasets/housing_test.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

# DIFFERENT K VALUES
k_vals = [1, 3, 5, 7, 9, 11, 13, 15]
mses = []

for k in k_vals:
    print(k)
    KNN = KNNClassifier(label_type='regression', weight_type='inverse_distance', k=k)
    KNN.fit(train_data_normalized, train_labels)
    mses.append(KNN.score(test_data_normalized, test_labels))

x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, mses)
ax.set_ylabel('Test MSE')
ax.set_title('Test MSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()


# Part 5
mat = Arff("datasets/credit_approval.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
data = raw_data[:,:-1]
labels = raw_data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

x_min = np.nanmin(X_train, axis=0)
x_max = np.nanmax(X_train, axis=0)
X_train_normalized = (X_train - x_min) / (x_max - x_min)
X_test_normalized = (X_test - x_min) / (x_max - x_min)

KNN = KNNClassifier(label_type='classification', weight_type='inverse_distance', distance_type='heom',
                    attr_types=mat.attr_types, attr_ranges=(x_min, x_max))
KNN.fit(X_train_normalized, y_train)
score = KNN.score(X_test_normalized, y_test)
print("CREDIT APPROVAL DATASET")
print("Accuracy = " + str(score))
print()


# Part 6
# MAGIC TELESCOPE
mat = Arff("datasets/magic_telescope_train.arff", label_count=1)
mat2 = Arff("datasets/magic_telescope_test.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree'],
              'p': [1, 2]}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, return_train_score=True)
grid_search.fit(train_data_normalized, train_labels)

print("Best score: {:.4f}".format(grid_search.best_score_))
print("Best parameters: {}".format(grid_search.best_params_))

bestknn = KNeighborsClassifier(algorithm='auto', n_neighbors=7, p=1, weights='distance')
bestknn.fit(train_data_normalized, train_labels)

print("Best test accuracy: " + str(bestknn.score(test_data_normalized, test_labels)))

# HOUSING
mat = Arff("datasets/housing_train.arff", label_count=1)
mat2 = Arff("datasets/housing_test.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

x_min = np.amin(train_data, axis=0)
x_max = np.amax(train_data, axis=0)
train_data_normalized = (train_data - x_min) / (x_max - x_min)
test_data_normalized = (test_data - x_min) / (x_max - x_min)

param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree'],
              'p': [1, 2]}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10, return_train_score=True)
grid_search.fit(train_data_normalized, train_labels)

print("Best parameters: {}".format(grid_search.best_params_))

bestknn = KNeighborsRegressor(algorithm='auto', n_neighbors=3, p=1, weights='distance')
bestknn.fit(train_data_normalized, train_labels)
pred = bestknn.predict(test_data_normalized)
mse = mean_squared_error(test_labels, pred)

print("Best MSE: " + str(mse))


# Part 7
# BODYFAT - ALL FEATURES
mat = Arff("datasets/bodyfat.arff", label_count=1)
raw_data = mat.data
h,w = raw_data.shape
data = raw_data[:,2:-1]
labels = raw_data[:,1]

x_min = np.amin(data, axis=0)
x_max = np.amax(data, axis=0)
data_normalized = (data - x_min) / (x_max - x_min)

X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.25)

knn = KNeighborsRegressor(algorithm='auto', n_neighbors=3, p=1, weights='distance')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("MSE with all features: " + str(mse))

# BODYFAT - PCA
print("Original num features: " + str(np.shape(X_train)[1]))
pca = PCA(0.95)
principal_components = pca.fit(X_train)
print("Num components that explain 95% of variance: " + str(pca.n_components_))

X_train_reduced = principal_components.transform(X_train)
X_test_reduced = principal_components.transform(X_test)

knn = KNeighborsRegressor(algorithm='auto', n_neighbors=3, p=1, weights='distance')
knn.fit(X_train_reduced, y_train)
pred = knn.predict(X_test_reduced)
mse = mean_squared_error(y_test, pred)

print("MSE with PCA features: " + str(mse))

# BODYFAT - FORWARD WRAPPER
best_mse = float('inf')

feature_indices = []
remaining_indices = set(np.arange(0, np.shape(X_train)[1]))
done = False

while not done:
    for i in remaining_indices:
        curr_indices = feature_indices.copy()
        curr_indices.append(i)

        curr_training_set = X_train[:, curr_indices]
        curr_test_set = X_test[:, curr_indices]

        knn = KNeighborsRegressor(algorithm='auto', n_neighbors=3, p=1, weights='distance')
        knn.fit(curr_training_set, y_train)
        pred = knn.predict(curr_test_set)
        mse = mean_squared_error(y_test, pred)

        if mse < best_mse:
            best_mse = mse
            feature_indices.append(i)
            remaining_indices.remove(i)
            break

        else:
            done = True
            break

print("Num features with forward wrapper: " + str(len(feature_indices)))
print("MSE with forward wrapper: " + str(best_mse))
