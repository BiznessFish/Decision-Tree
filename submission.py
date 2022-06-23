import numpy as np
from typing import List, Tuple, Iterable, Optional

class Node:
    def __init__(self,
    n_items: int, 
    ATE: float,
    split_feat: int = None,
    split_threshold: Optional[float] = None,
    isleaf: bool = False,
    left: Optional['Node'] = None,
    right: Optional['Node'] = None
    ):
        """Initializes helper class Node to create a UpliftTreeRegresor
        """

        self.n_items = n_items
        self.ATE = ATE
        self.split_feat = split_feat
        self.split_threshold = split_threshold
        self.isleaf = isleaf
        self.left = left
        self.right = right
    
    def _print_self(self, 
        depth:int = 0) -> None:
        """Internal method to print out the whole tree
        """

        if depth == 0:
            print('Root')

        # print properties of each node
        print('\t'*depth + "n_items: ", self.n_items)
        print('\t'*depth + "ATE: ", self.ATE)
        print('\t'*depth + "split_feat: feat",self.split_feat)
        print('\t'*depth + "split_threshold: ", self.split_threshold)
        print('\n')

        # stop when you hit a leaf
        if self.isleaf:
            return

        #add a lil somethin if the node is a leaf
        if self.left.isleaf :
            print('\t'*(depth+1) + "Left <Leaf>")
        else:
            print('\t'*(depth+1) + "Left")        
        self.left._print_self(depth + 1)

        if self.right.isleaf :
            print('\t'*(depth+1) + "Right <Leaf>")
        else:
            print('\t'*(depth+1) + "Right")        
        self.right._print_self(depth + 1)

    def _predict(self, X: np.ndarray) -> Iterable[float]:
        """Internal method to make predictions by traversing tree. 
        Called by UpliftTreeRegressor class (and itself)
        """

        if X.ndim > 1:
            return np.array([self._predict(x) for x in X])
        
        # recurse until a leaf is hit
        if self.isleaf:
            return self.ATE
        else:
            if X[self.split_feat] <= self.split_threshold:
                return self.left._predict(X)
            else:
                return self.right._predict(X)
        
class UpliftTreeRegressor:

    def __init__(
        self,
        max_depth: int =3, # max tree depth
        min_samples_leaf: int = 1000, # min number of values in leaf
        min_samples_leaf_treated: int = 300, # min number of treatment values in leaf
        min_samples_leaf_control: int = 300, # min number of control values in leaf 
        ):
        """Initializes a UpliftTreeRegressor class. 

        Attributes:
            max_depth: int =3, # max tree depth
            min_samples_leaf: int = 1000, # min number of values in leaf
            min_samples_leaf_treated: int = 300, # min number of treatment values in leaf
            min_samples_leaf_control: int = 300, # min number of control values in leaf 

        """

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def fit(
        self,
        X: np.ndarray, # (n * k) array with features
        Treatment: np.ndarray, # (n) array with treatment flag
        Y: np.ndarray # (n) array with the target
        ) -> None:
        """This method fits a decision tree, splitting on the best deltadeltaP criterion.

        Args:
            X : a numpy ndarray (n * k) with features. 
            Treatment: a numpy array (n) with treatment flag of 0 or 1
            Y : a numpy array with the response variable. 

        Returns: 
            None
        
        """

        self._root = self._build(X, Treatment, Y)

    def predict(self, X: np.ndarray) -> Iterable[float]:
        """This method returns predictions made by a fitted decision tree. Input 
        must have the same dimension k (# of features) as the training data.

        Args:
            X : a numpy ndarray (n * k) to be predicted

        Returns: 
            np.ndarray of floats
        
        """
        return self._root._predict(X)

    def _get_threshold_values(self, 
        column_values: np.ndarray # column_values - 1d array with the feature values in current node
        ) -> np.ndarray : # threshold_options - threshold values to go over to find the best one
        """
        Internal method for calculating threshold values
        """
        
        unique_values = np.unique(column_values)
        if len(unique_values) >10:
            percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        
        threshold_options = np.unique(percentiles)

        return threshold_options

    def _get_ATE(self, 
        Treatment: np.ndarray,
        Y: np.ndarray
        ) -> float:
        """
        Internal method for calculating average treatment effect
        """

        treated_Y = Y[Treatment.astype(bool)]
        untreated_Y = Y[~Treatment.astype(bool)]

        # calculate ATE as simple difference in mean. Does NOT take into account selection bias
        # or heterogenous treatment effects since we don't know anything about the features
        ATE = treated_Y.mean() - untreated_Y.mean()
    
        return ATE

    def _split(self,
            X: np.ndarray, 
            Treatment: np.ndarray,
            Y: np.ndarray
            ) -> Tuple[int, float, np.ndarray]:
        """Internal method to get split, feature index, and a mask for the split
        """
        
        best_feature_index = np.nan
        best_split_value = np.nan
        best_mask = None
        best_DDP = 0

        for feature_index, feature in enumerate(X.T):

            # get split thresholds
            threshold_values = self._get_threshold_values(feature)
            for value in threshold_values:

                mask = np.ravel(feature < value)
                left_X = X[mask]
                left_Y = Y[mask]
                left_treatment = Treatment[mask]

                right_X = X[~mask]
                right_Y = Y[~mask]
                right_treatment = Treatment[~mask]

                # see if it fails any checks; discard the split value if it does

                is_minleaf_fail = (len(left_Y) < self.min_samples_leaf) or \
                                    (len(right_Y) < self.min_samples_leaf) 
                
                is_mintreated_fail = (sum(left_treatment) < self.min_samples_leaf_treated) or \
                                        (sum(right_treatment) < self.min_samples_leaf_treated)

                is_mincontrol_fail = (len(left_treatment) - sum(left_treatment) < self.min_samples_leaf_control) or \
                                        (len(right_treatment) - sum(right_treatment) < self.min_samples_leaf_control) 
                
                checks = [is_minleaf_fail, is_mintreated_fail, is_mincontrol_fail]

                if any(checks):
                    continue

                # get DDP
                left_ATE = self._get_ATE(left_treatment, left_Y)
                right_ATE = self._get_ATE(right_treatment, right_Y)

                DDP = abs(left_ATE - right_ATE)

                if DDP > best_DDP:
                    best_feature_index = feature_index
                    best_split_value = value
                    best_DDP = DDP
                    best_mask = mask


        return best_feature_index, best_split_value, best_mask

    def _build(self, 
        X: np.ndarray, 
        Treatment: np.ndarray,
        Y: np.ndarray,
        depth: int = 0 ) -> Node:
        """Internal method to grow decision tree recursively
        """
                
        # If node is at max depth, stop and make a leaf node.
        if depth == self.max_depth:
            return Node(len(Y), self._get_ATE(Treatment, Y), isleaf=True)

        # Get the feature index, split value, and mask
        feature_index, split_value, mask = self._split(X, Treatment, Y)

        # if theres no split that works, then end at a leaf
        if np.isnan(split_value):
            return Node(len(Y), self._get_ATE(Treatment, Y), isleaf=True)        

        return Node( n_items = len(Y) , 
                    ATE = self._get_ATE(Treatment, Y), 
                    split_feat =feature_index, 
                    split_threshold= split_value ,                     
                        left= self._build(X = X[mask], Treatment = Treatment[mask], Y = Y[mask], depth = depth + 1) , 
                        right= self._build(X = X[~mask], Treatment = Treatment[~mask], Y = Y[~mask], depth = depth + 1) )
