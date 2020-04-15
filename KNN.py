import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, label_type='classification', weight_type='inverse_distance', distance_type='euclidean', attr_types=None, attr_ranges=None, k=3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.column_type = label_type
        self.weight_type = weight_type
        self.distance_type = distance_type
        self.attr_types = attr_types
        self.attr_ranges = attr_ranges
        self.k = k

        if self.distance_type == 'heom':
            assert self.attr_types is not None and self.attr_ranges is not None

    def fit(self, data, labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = data
        self.labels = labels

        return self

    # Helper function for calculating the HEOM distance metric
    def _heom_distance(self, data_point, curr_instance):
        d_a = 0

        for i in range(np.shape(data_point)[1]):
            if np.isnan(data_point[0, i]) or np.isnan(curr_instance[0, i]):
                d_a += 1

            elif self.attr_types[i] == 'nominal':
                d_a += 0 if data_point[0, i] == curr_instance[0, i] else 1

            else:
                d_a += (abs(data_point[0, i] - curr_instance[0, i]) / (self.attr_ranges[1][i] - self.attr_ranges[0][i])) ** 2

        return d_a ** 0.5

    def knn(self, data_point):
        n_instances = np.shape(self.data)[0]
        distances = np.zeros(n_instances)
        weighted_distances = np.zeros(n_instances)

        for i in range(n_instances):
            if self.distance_type == 'heom':
                distance = self._heom_distance(data_point, self.data[i, :].reshape(1, -1))

            else:
                distance = np.sum((data_point - self.data[i, :].reshape(1, -1)) ** 2, axis=1) ** 0.5

            distances[i] = distance
            weighted_distances[i] = 1 / (distance ** 2)

        nearest_indices = np.argsort(distances)

        if self.weight_type == 'inverse_distance':
            class_weighted_distances = {}

            for i in nearest_indices[:self.k]:
                curr_class = self.labels[i]
                class_weighted_distances[curr_class] = class_weighted_distances.get(curr_class, 0) + weighted_distances[i]

            closest = max(class_weighted_distances, key=class_weighted_distances.get)

        else:
            class_counts = {}

            for i in nearest_indices[:self.k]:
                curr_class = self.labels[i]
                class_counts[curr_class] = class_counts.get(curr_class, 0) + 1

            closest = max(class_counts, key=class_counts.get)

        return closest

    def regression_knn(self, data_point):
        n_instances = np.shape(self.data)[0]
        distances = np.zeros(n_instances)

        for i in range(n_instances):
            if self.distance_type == 'heom':
                distance = self._heom_distance(data_point, self.data[i, :].reshape(1, -1))

            else:
                distance = np.sum((data_point - self.data[i, :].reshape(1, -1)) ** 2, axis=1) ** 0.5

            distances[i] = distance

        nearest_indices = np.argsort(distances)
        regression_output = 0
        weighted_distance = 0

        for i in nearest_indices[:self.k]:
            curr_class = self.labels[i]
            regression_output += curr_class / (distances[i] ** 2) if self.weight_type == 'inverse_distance' else curr_class
            weighted_distance += 1 / (distances[i] ** 2)

        return regression_output / weighted_distance if self.weight_type == 'inverse_distance' else regression_output / self.k

    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        output = []

        for i in range(np.shape(data)[0]):
            if self.column_type == 'classification':
                prediction = self.knn(data[i, :].reshape(1, -1))

            else:
                prediction = self.regression_knn(data[i, :].reshape(1, -1))

            output.append(prediction)

        return output

    #Returns the Mean score given input data and labels
    def score(self, X, y):
            """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                    X (array-like): A 2D numpy array with data, excluding targets
                    y (array-like): A 2D numpy array with targets
            Returns:
                    score : float
                            Mean accuracy of self.predict(X) wrt. y.
            """
            # Use our predict method
            output = self.predict(X)
            num_instances = np.shape(X)[0]

            if self.column_type == 'classification':
                # Find the total number of correct predictions and the total number of data instances
                num_correct = np.sum(np.all(np.array(output).reshape(-1, 1) == np.array(y).reshape(-1, 1), axis=1))

                accuracy = num_correct / num_instances

            else:
                accuracy = np.sum((y - output) ** 2) / num_instances

            # Calculate and return the accuracy
            return accuracy


