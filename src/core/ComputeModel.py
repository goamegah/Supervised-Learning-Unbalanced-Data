import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.core.Processing import Processing
from src.core.Model import Model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import zero_one_loss
from src.core.globals import PARAMS_GRID
from mlxtend.evaluate import bias_variance_decomp

class ComputeModel:

    def __init__(
            self,
            df_X: pd.DataFrame,# Feature dataframe
            y: pd.Series, # Target variable
            model_name: str, # Name of the model
            process: Processing, # Processing object
            hyperparameters: dict = None, # Hyperparameters for the model
            l_order: list = None,# List of ordinal columns
            positive_mod=None, # Positive class
            grid_search=False, # Whether to perform grid search or not
            random_search=False, # Whether to perform random search or not
            stratify=True,  # Whether to stratify or not
            params_grid=None,   # Parameters grid
            cv=3, # Number of cross-validation folds
            verbose=1
    ):
        self.df_X = df_X
        self.y = y
        self.search = grid_search or random_search
        self.stratify = stratify
        # Initializing the model attribute based on whether search (tuning) is to be performed or not
        if not self.search:
            self.model = Model(model_name, hyperparameters=hyperparameters)
        else:
            # Setting the parameters grid if not provided
            self.params_grid = PARAMS_GRID[model_name] if params_grid is None else params_grid
            cv = cv if not stratify else StratifiedKFold(cv)
            # Initializing the search object based on whether random search is to be performed or not
            if random_search:
                search = RandomizedSearchCV(Model(model_name).model, param_distributions=self.params_grid, cv=cv,
                                            verbose=verbose)
            else:
                search = GridSearchCV(Model(model_name).model, self.params_grid, verbose=verbose, cv=cv)
            self.model = Model(model_name, search=search)
        self.process = process
        self.l_order = l_order
        modalities_y = pd.unique(self.y)
        # Setting the positive_mod attribute
        if positive_mod is None:
            positive_mod = modalities_y[0]
        if not (positive_mod in modalities_y):
            raise Exception(f"{positive_mod} should be modality of {y}")
        self.positive_mod = positive_mod

    def numerize(self, df: pd.DataFrame) -> dict:
        # Converting the feature dataframe df to a numerical dataframe
        if self.l_order is not None:
            result = self.process.df_to_numerical(df, l_order=self.l_order)
            df = result["df_transform"]
            mapping = result["mapping"]
            return {"df_transform": df, "mapping": mapping}
        else:
            df = self.process.df_to_numerical(df, l_order=self.l_order)["df_transform"]
            return {"df_transform": df}

    def split(self, test_size=.2, random_state=42,method="SMOTE",sampling=True,perc_minority=.5) -> None:
        """
        Defining a method 'split' which splits the data into train and test sets
        The method takes several parameters including test_size, random_state, sampling method, sampling and perc_minority
        """
        # Splitting the data into train and test sets
        df_to_np = self.numerize(self.df_X)
        # Converting the dataframe to numpy array
        X = np.array(df_to_np["df_transform"])
        # If mapping is present, extract it from the converted data
        mapping = df_to_np["mapping"] if "mapping" in df_to_np.keys() else None
        # Extracting the target variable and converting it to a binary output
        y = np.array(self.y == self.positive_mod).reshape(-1)
        # Checking if the sampling parameter is set to True
        if sampling == True:
            # Checking which sampling method to use
            if method == "SMOTE":
                # Oversampling the data using the SMOTE method
                X,y=self.process.oversampling(X,y,perc_minority="auto" if perc_minority==.5 else perc_minority)
            elif method == "RandomUnderSampling":
                # Undersampling the data using the RandomUnderSampling method
                X,y=self.process.undersampling(X,y,perc_minority=perc_minority)

        # Checking if stratification is required while splitting the data
        stratify = y if self.stratify else None
        # Splitting the data into train and test sets using the train_test_split method from sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=stratify)
        # Creating a dictionary to store the split data
        if self.l_order:
            self.dict_split = {
                "arrays": {
                    "X_train": X_train,
                    "y_train": y_train.reshape(-1),
                    "X_test": X_test,
                    "y_test": y_test.reshape(-1)
                },
                "columns_features": df_to_np["df_transform"].columns,
                "mapping": mapping
            }
        else:
            # case: qualitatives variable are all nominal (no ordinal variable)
            self.dict_split = \
                {
                    "arrays": {
                        "X_train": X_train,
                        "y_train": y_train.reshape(-1),
                        "X_test": X_test,
                        "y_test": y_test.reshape(-1)
                    },
                    "columns_features": df_to_np["df_transform"].columns
                }

    def fit(self, test_size=.2, random_state=42,sampling=False,method="SMOTE",perc_minority=.5) -> None:
        """
        This method fits the model to the training data
        It splits the data into training and test sets using the specified test_size and random_state
        If sampling is True, it applies SMOTE oversampling to the minority class (by default)
        It then fits the model to the training data, either using the default fit method or a search method if self.search is True
        Finally, it updates the model if necessary (in case of gridsearch/randomsearch)
        """
        self.split(test_size, random_state,method=method,sampling=sampling,perc_minority=perc_minority)
        X_train = self.dict_split["arrays"]["X_train"]
        y_train = self.dict_split["arrays"]["y_train"]
        if not self.search:
            #no tuning of hyperparameters
            self.model.fit(X_train, y_train)
        else:
            #tuning of hyperparameters
            self.model.search.fit(X_train, y_train)
            self.model.update()

    def metrics(self, plot_roc=False, ax: Axes = None):
        """
        This method calculates the performance metrics of the model on the test set
        It raises an exception if the train/test set is not defined
        It retrieves the test set from the stored dictionary and passes it to the model's metrics method
        It returns the result of the metrics method
        if plot_roc=True: roc_curve are set up in ax attribute of the method
        """
        if not hasattr(self,"dict_split"):
            raise Exception("a train/test set should defined.")
        X, y = self.dict_split["arrays"]["X_test"], self.dict_split["arrays"]["y_test"]
        return self.model.metrics(X, y.reshape(-1), plot_roc=plot_roc, ax=ax)

    def predict_proba(self, df: pd.DataFrame):
        """
        This method predicts the probability of each class for the given input data
        It applies the numerical transformation to the input data using the numerize method
        It then passes the transformed data to the model's predict_proba method
        """
        return self.model.predict_proba(np.array(self.numerize(df)["df_transform"]))




    def permutation_importance_model(self, scoring="roc_auc"):
        """
        This method calculates the permutation importance of each feature in the training data
        It passes the training data and the specified scoring metric to the model's permutation_importance_model method
        """
        return self.model.permutation_importance_model(
            self.dict_split["arrays"]["X_train"],
            self.dict_split["arrays"]["y_train"],
            scoring
        )



    def bias_variance_estimate(self,bootstrap_rounds = 100):
        """
        This method estimates the bias and variance of the model using the bootstrap method
        It splits the data into training and test sets using the specified method, sampling, and perc_minority
        It initializes a dataframe for storing the predictions on the test data
        For each round, it draws bootstrap indices, trains the model on the bootstrap data, and makes predictions on the test data
        It calculates the "average model"'s predictions and then calculates the bias and variance using the zero_one_loss metric
        It returns the calculated bias and variance
        """

        self.split(.2, 42,method="SMOTE",sampling=True,perc_minority="auto")
        estimator=self.model.model
        X_train,y_train= pd.DataFrame(self.dict_split["arrays"]["X_train"]),pd.DataFrame(self.dict_split["arrays"]["y_train"].reshape(-1))
        X_test,y_test= pd.DataFrame(self.dict_split["arrays"]["X_test"]),pd.DataFrame(self.dict_split["arrays"]["y_test"].reshape(-1))

        # initialize dataframe for storing predictions on test data
        preds_test = pd.DataFrame(index = y_test.index)
        # for each round: draw bootstrap indices, train model on bootstrap data and make predictions on test data
        for r in range(bootstrap_rounds):
            boot = np.random.randint(len(y_train), size = len(y_train))
            preds_test[f'Model {r}'] = estimator.fit(np.array(X_train.iloc[boot, :]), np.array(y_train.iloc[boot]).ravel()).predict(X_test)
        # calculate "average model"'s predictions
        mean_pred_test = preds_test.mean(axis = 1) >= .5

        bias_squared = zero_one_loss(y_test, mean_pred_test)
        variance = preds_test.apply(lambda pred_test: zero_one_loss(mean_pred_test, pred_test)).mean()
        return bias_squared, variance