from typing import Literal
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class TreeNode:
    model: LinearRegression = None
    mark: Literal[1, -1] = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None


class MultivariateDecisionTree:

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: TreeNode = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(x, y, 0)

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int):
        n_samples = len(y)
        n_labels = len(np.unique(y))

        # end condition
        if n_labels == 0:
            return TreeNode(mark=-1)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return TreeNode(mark=self._majority_vote(y))

        # find the best split
        model = LinearRegression()
        model.fit(x, y)
        pred = model.predict(x)
        left = self._build_tree(x[pred >= 0.5], y[pred >= 0.5], depth + 1)
        right = self._build_tree(x[pred < 0.5], y[pred < 0.5], depth + 1)
        return TreeNode(model=model, left=left, right=right)

    def _majority_vote(self, y: np.ndarray):
        return np.argmax(np.bincount(y))

    def nodes(self) -> list[TreeNode]:
        return self._nodes(self.root)

    def _nodes(self, node: TreeNode):
        if node is None:
            return []
        return [node] + self._nodes(node.left) + self._nodes(node.right)


if __name__ == "__main__":
    watermelon_data = np.array([
        [0.697, 0.460],
        [0.774, 0.376],
        [0.634, 0.264],
        [0.608, 0.318],
        [0.556, 0.215],
        [0.403, 0.237],
        [0.437, 0.211],
        [0.666, 0.091],
        [0.243, 0.267],
        [0.245, 0.057],
        [0.343, 0.099],
        [0.639, 0.161],
        [0.657, 0.198],
        [0.360, 0.370],
        [0.593, 0.042],
        [0.719, 0.103]
    ])
    watermelon_labels = np.array([
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    assert len(watermelon_data) == len(watermelon_labels)

    tree = MultivariateDecisionTree()
    tree.fit(watermelon_data, watermelon_labels)

    import matplotlib.pyplot as plt

    # draw the data point
    for i in range(len(watermelon_data)):
        if watermelon_labels[i] == 1:
            plt.scatter(watermelon_data[i][0], watermelon_data[i][1], color='red')
        else:
            plt.scatter(watermelon_data[i][0], watermelon_data[i][1], color='blue')

    # draw the decision boundary
    for node in tree.nodes():
        if node.model is not None:
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 100))
            z = node.model.predict(np.c_[xx.ravel(), yy.ravel()])
            z = z.reshape(xx.shape)
            plt.contour(xx, yy, z, [0.5])

    plt.xlim(0, 0.8)
    plt.ylim(0, 0.6)
    plt.show()
