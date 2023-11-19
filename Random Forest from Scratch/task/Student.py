import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import trange

np.random.seed(52)


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.forest = []
        self.is_fit = False

    def fit(self, X_train, y_train):
        for _ in trange(self.n_trees):
            x_s, y_s = self.create_bootstrap(X_train, y_train)
            tree = DecisionTreeClassifier(max_features='sqrt', max_depth=self.max_depth,
                                          min_impurity_decrease=self.min_error, random_state=52)
            tree.fit(x_s, y_s)
            self.forest.append(tree)
        self.is_fit = True
        return self

    def predict(self):
        if not self.fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

    @staticmethod
    def create_bootstrap(bx, by):
        mask = np.random.choice(bx.shape[0], bx.shape[0], replace=True)
        return bx[mask], by[mask]


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    forest = RandomForestClassifier().fit(X_train, y_train)
    prediction = forest.forest[0].predict(X_val)
    print(round(accuracy_score(y_val, prediction), 3))
