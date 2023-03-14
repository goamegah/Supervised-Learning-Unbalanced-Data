from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from src.globals import MODEL_HYPERPARAMETERS_DEF
from sklearn.model_selection import GridSearchCV


class Model:

    def __init__(
            self,
            model_name: str,
            hyperparameters: dict = None,
            grid_search: GridSearchCV = None
    ):
        self.model_name = model_name
        if grid_search is None:
            if hyperparameters is None:
                hyperparameters = MODEL_HYPERPARAMETERS_DEF[self.model_name]
            self.hyperparameters = hyperparameters
            if model_name == "Logistic Regression":
                self.model = LogisticRegression(**self.hyperparameters)
            elif model_name == "SVM":
                self.model = SVC(**self.hyperparameters)
            elif model_name == "DecisionTreeClassifier":
                self.model = DecisionTreeClassifier(**self.hyperparameters)
        else:
            self.model = None
            self.hyperparameters = None
            self.grid_search = grid_search

    def update(self):
        """
        :return: update model only for GridSearchCV
        """
        self.model = self.grid_search.best_estimator_
        self.hyperparameters = self.grid_search.best_params_

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        if self.model_name == "SVM":
            dist_to_hyperplan = self.model.decision_function(X)
            try:
                return (dist_to_hyperplan - dist_to_hyperplan.min()) / (
                        dist_to_hyperplan.max() - dist_to_hyperplan.min())
            except ZeroDivisionError:
                print(f"A problem occur during min-max normalization for testset:{X}")
        elif self.model_name == "Logistic Regression" or self.model_name == "DecisionTreeClassifier":
            return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def metrics(self, X, y, plot_roc=False, ax: Axes = False):
        precision, recall, f1_score, support = precision_recall_fscore_support(y, self.predict(X), average="binary",
                                                                               pos_label=True)
        fpr, tpr, thresholds = roc_curve(y, self.predict_proba(X))
        auc_value = auc(fpr, tpr)
        if plot_roc:
            if ax is None:
                raise Exception(f"{ax} for plotting roc curve is None")
            ax.plot(fpr, tpr)
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            ax.set_title(f"Roc Curve for {self.model_name}")
        return {
            "auc": auc_value,
            "accuracy": self.model.score(X, y),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def plot_model(self):
        if self.model_name == "DecisionTreeClassifier":
            return plot_tree(self.model)
