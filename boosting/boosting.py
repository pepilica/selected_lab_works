from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        new_model = self.base_model_class(**self.base_model_params)
        bootstrapped_indices = np.random.choice(np.array(range(y.shape[0])), int(self.subsample * y.shape[0]), replace=True)
        y_train = y[bootstrapped_indices]
        x_train = x[bootstrapped_indices, :]
        y_pred = predictions[bootstrapped_indices]
        s = -self.loss_derivative(y_train, y_pred)
        new_model = new_model.fit(x_train, s)
        new_gamma = self.find_optimal_gamma(y_train, y_pred, new_model.predict(x_train))
        self.gammas.append(self.learning_rate * new_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        loss_train = []
        loss_val = []

        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions = np.round(self.predict_proba(x_train)[:, 1])
            valid_predictions = np.round(self.predict_proba(x_valid)[:, 1])
            loss_train.append(self.loss_fn(y_train, train_predictions))
            loss_val.append(self.loss_fn(y_valid, valid_predictions))
            if self.early_stopping_rounds is not None:
                if i > self.early_stopping_rounds and loss_val[-1] > loss_val[-2]:
                    break
        if self.plot:
            plt.plot(range(len(loss_train)), loss_train, label='Loss on train set')
            plt.plot(range(len(loss_val)), loss_val, label='Loss on validational set')
            plt.legend()
            plt.xlabel("Number of built model")
            plt.ylabel("Loss")
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += gamma * model.predict(x)
        return np.hstack([ (antisigm := self.sigmoid(-predictions).reshape((-1, 1))), 1 - antisigm])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        return np.hstack([i.feature_importances_.reshape((-1, 1)) for i in self.models]) @ np.array(self.gammas).reshape((-1, 1)) / np.sum(self.gammas)
