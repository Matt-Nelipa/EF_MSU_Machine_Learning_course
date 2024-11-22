import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from collections import Counter
from category_encoders import TargetEncoder

from typing import List, Union, Tuple



def compute_criterion(target_vector: np.array, feature_vector: np.array, threshold: float, criterion: str = 'gini') -> float:
    """
    Calculates the criterion for splitting a node
        Q = H(R) - |R_l|/|R| * H(R_l) - |R_r|/|R| * H(R_r)

    Predicate for split: [feature_vector < threshold]

    If there is only 1 unique value in feature_vector, the function returns 0

    Args:
        target_vector: target vector (multiclass)
        feature_vector: vector with a specific object feature (real)
        threshold: threshold for splitting into left and right subtrees
        criterion: impurity criterion ("gini" or "entropy")
    Returns:
        Q: the splitting criterion, maximizing which we choose the optimal leaf partitioning
    """

    assert criterion in ['gini', 'entropy'], "Criterion must be 'gini' or 'entropy'!"

    r_leaf = []
    l_leaf = []
    H_R_r = 0
    H_R_l = 0
    H_R = 0
    len_target = len(target_vector)

    for i in range(len(target_vector)):
        if feature_vector[i] < threshold:
            l_leaf.append(target_vector[i])
        else:
            r_leaf.append(target_vector[i])

    if criterion == "gini":
        
        for class_ in np.unique(target_vector):
            H_R += np.count_nonzero(target_vector == class_) / len_target * (1 - (np.count_nonzero(target_vector == class_) / len_target))
    
        for gini_r in np.unique(r_leaf):
            prob_r = r_leaf.count(gini_r) / len(r_leaf)
            H_R_r += prob_r * (1 - prob_r) 

        for gini_l in np.unique(l_leaf):
            prob_l = l_leaf.count(gini_l) / len(l_leaf)
            H_R_l += prob_l * (1 - prob_l) 

    elif criterion == "entropy":
        for class_ in np.unique(target_vector):
            H_R -= np.count_nonzero(target_vector == class_) / len_target * np.log2(np.count_nonzero(target_vector == class_) / len_target)

        for ent_r in np.unique(r_leaf):
            ent_class_prob_r = r_leaf.count(ent_r) / len(r_leaf)
            H_R_r -= ent_class_prob_r * np.log2(ent_class_prob_r) 

        for ent_l in np.unique(l_leaf):
            ent_class_prob_l = l_leaf.count(ent_l) / len(l_leaf)
            H_R_l -= ent_class_prob_l * np.log2(ent_class_prob_l)
    

    Q = H_R - (len(l_leaf) / len_target * H_R_l) - (len(r_leaf) / len_target * H_R_r)
    
    return Q


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray, criterion: str = 'gini') -> Tuple:
    """
    Функция, находящая оптимальное разбиение с точки зрения критерия gini или entropy

    Args:
        feature_vector: вещественнозначный вектор значений признака
        target_vector: вектор классов объектов (многоклассовый),  len(feature_vector) == len(target_vector)
    Returns:
        thresholds: (np.ndarray) отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
                     разделить на две различные подвыборки, или поддерева
        criterion_vals: (np.ndarray) вектор со значениями критерия Джини/энтропийного критерия для каждого из порогов
                в thresholds. # len(criterion_vals) == len(thresholds)
        threshold_best: (float) оптимальный порог
        criterion_best: (float) оптимальное значение критерия

    A function that finds the optimal partition in terms of the gini or entropy criterion
    Args:
        feature_vector: a real-valued vector of feature values
        Target_vector: vector of feature classes (multi-class), len(feature_vector) == len(target_vector)
    Returns:
        thresholds: (np.ndarray) an ascending sorted vector with all possible thresholds 
        by which features can be into two different subsamples, or subtrees
        criterion_vals: (np.ndarray) a vector with the values of the Gini/entropic criterion 
        for each of the thresholds # len(criterion_vals) == len(thresholds)
        threshold_best: (float) optimal threshold
        criterion_best: (float) optimal criterion value
    """

    vals_unique = np.sort(np.unique(feature_vector))

    if len(vals_unique) == 1:
        return None, None, None, 0

    thresholds = np.array([])
    criterion_vals = np.array([])
    threshold_best = None
    criterion_best = 0

    possible_thresholds = []
    for i in range(len(vals_unique) - 1):
        possible_thresholds.append(np.mean([vals_unique[i], vals_unique[i+1]]))
    

    for threshold in possible_thresholds:
        Q = compute_criterion(target_vector, feature_vector, threshold, criterion)
        thresholds = np.append(thresholds, threshold) 
        criterion_vals = np.append(criterion_vals, Q)
    
    zipped = list(zip(criterion_vals, thresholds))
    sorted_data = sorted(zipped, key=lambda x: (-x[0], x[1]))
    criterion_best, threshold_best = sorted_data[0]

    return (thresholds, criterion_vals, threshold_best, criterion_best)



class DecisionTree(BaseEstimator):

    def __init__(
            self,
            feature_types: list,
            criterion: str = 'gini',
            max_depth: int = None,
            min_samples_split: int = None,
    ):
        """
        Args:
            feature_types: list of feature types (can be 'real' and 'categorical')
            criterion: can be 'gini' or 'entropy'
            max_depth: maximum depth of the tree
            min_samples_split: the minimum number of objects in a leaf to be able to split the leaf.
        """

        self._feature_types = feature_types
        self._tree = {}
        self.target_encodings = {}
        self._criterion = criterion
        self.max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._y_unique = None 
        self._current_depth = 1

        
    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: dict):
        """
        Seeks the optimal split for a leaf consisting of sub_X objects and sub_y targets. 
        If for this leaf break criteria are met, 
        it terminates and designates the leaf type as terminal (type=“terminal”).
        
        In case the feature is of type “categorical”, target encoding is applied to it 
        and the trained encoding is written to self.target_encodings

        Args:
            sub_X: array of size (n, len(self._feature_types)), the object-attribute matrix 
            for the objects in the current sheet
            sub_y: array of size (n,), a vector of targets for the objects hit in the current leaf
            node: dictionary containing the tree trained to the current leaf
        Returns:
                None
        """

        best_thresholds_for_every_feature = []
        best_Q_for_every_feature = []


        

        for i in range(sub_X.shape[1]):

            column_result = find_best_split(sub_X[:,i], sub_y, self._criterion)
            best_thresholds_for_every_feature.append(column_result[2])
            best_Q_for_every_feature.append(column_result[3])


        best_Q_index = best_Q_for_every_feature.index(max(best_Q_for_every_feature))
        best_threshold = best_thresholds_for_every_feature[best_Q_index]
        
        if best_threshold == None:
            best_threshold = 1

        concat_X_y = np.concatenate([sub_X, sub_y.reshape(-1, 1)], axis=1)
        X_y_r = np.array([concat_X_y[i, :] for i in range(concat_X_y.shape[0]) if concat_X_y[i, best_Q_index] >= best_threshold])
        X_y_l = np.array([concat_X_y[i, :] for i in range(concat_X_y.shape[0]) if concat_X_y[i, best_Q_index] < best_threshold])
        current_depth = self._current_depth
        y_classes_counter_sorted = list(dict(sorted({i: np.count_nonzero(sub_y == i) for i in list(self._y_unique)}.items())).values())
        

        if (current_depth < self.max_depth) and (sub_y.shape[0] >= self._min_samples_split if type(self._min_samples_split) == int else sub_y.shape[0] >= int(bool(self._min_samples_split)) and sub_y.shape[0] >= 2) and (len([num for num in y_classes_counter_sorted if num != 0]) > 1) and (X_y_l.shape[0] != 0 and X_y_r.shape[0] != 0):
            X_l, y_l = X_y_l[:, :-1], X_y_l[:, -1]
            node["type"] = "nonterminal"
            node['feature_type'] = self._feature_types[best_Q_index],
            node['feature_number'] = best_Q_index,
            node["threshold"] = best_threshold
            node['left_child'] = {}
            self._current_depth += 1
            self._fit_node(X_l, y_l, node['left_child']) 

        else:
            node["type"] = "terminal"
            classes_counts = np.array(y_classes_counter_sorted)
            node["classes_distribution"] = classes_counts
            return None
        
        self._current_depth = current_depth    
            
        if (current_depth < self.max_depth) and (sub_y.shape[0] >= self._min_samples_split if type(self._min_samples_split) == int else sub_y.shape[0] >= int(bool(self._min_samples_split)) and sub_y.shape[0] >= 2) and (len([num for num in y_classes_counter_sorted if num != 0]) > 1) and (X_y_l.shape[0] != 0 and X_y_r.shape[0] != 0):
            
            X_r, y_r = X_y_r[:, :-1], X_y_r[:, -1]
            node["type"] = "nonterminal"
            node['feature_type'] = self._feature_types[best_Q_index]
            node['feature_number'] = best_Q_index
            node["threshold"] = best_threshold
            node['right_child'] = {}
            self._current_depth += 1
            self._fit_node(X_r, y_r, node['right_child'])

        else:
            node["type"] = "terminal"
            classes_counts = np.array(y_classes_counter_sorted)
            node["classes_distribution"] = classes_counts

        self._current_depth = current_depth
        return None


    def _predict_proba_object(self, x: np.array, node: dict) -> Union[List, np.ndarray]:
        """
        Either returns the class distribution for object x,
        or recursively goes to the left or right subtree.
        Args:
            x: object of the size (len(self._feature_types),)
            node: trained tree that can be used to predict
        """
        # your code here:
        if node['type'] == 'nonterminal':
            if x[node["feature_number"]] < node['threshold']:
                return self._predict_proba_object(x, node['left_child'])
            else:
                return self._predict_proba_object(x, node["right_child"])
            
        else:
            return node["classes_distribution"] / sum(node["classes_distribution"])


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: object-feature matrix of the size (n, len(self._feature_types))
            y: target vector of the size (n,)
        """
        assert len(set(y)) > 1, 'Target must contain more than one class!'
        

        self.target_encodings = {col: TargetEncoder() for col in range(len(self._feature_types)) if self._feature_types[col] == "categorical"}

        for i in self.target_encodings.keys():
            self.target_encodings[i].fit(X[:, i], y)
            X[:, i] = np.array(self.target_encodings[i].transform(X[:, i])).ravel()

        self._y_unique = np.unique(y)
        y = y.reshape(-1, 1)
        self._fit_node(sub_X = X, sub_y = y, node=self._tree)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Applies self._predict_proba_node to each row from X
        Args:
            X: objects matrix of the size (m, len(self._feature_types)) for which to make prediction
        Returns:
            np.ndarray of the size (len(X), len(set(y)) (where y - target vector, used in the fit method (self.fit))
        """
        assert self._tree != {}, "First, train the model!"
        pred = []

        for col in self.target_encodings.keys():
            X[:, col] = np.array(self.target_encodings[col].transform(X[:, col])).ravel()

        for x in X:
            pred.append(self._predict_proba_object(x, self._tree))
        return np.array(pred)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X=X), axis=1).ravel()