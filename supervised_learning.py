import pandas as pd
import numpy as np

from lets_plot import *

# packages for Supervised Learning


from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.svm import SVR # Support Vector Machine

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.linear_model import ElasticNet # Elastic Net Penalty
from sklearn.linear_model import Lasso # LASSO


from sklearn.tree import DecisionTreeRegressor # Decision Tree


from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

import statsmodels.tsa.arima.model as stats
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_squared_error, accuracy_score

from feature_engineering import create_all_features


# ============================================================
# LABEL CONSTRUCTION
# ============================================================

def make_future_log_return(df, price_column="Close", horizon=1):
    """
    Y_t = log(Close_{t+h}) - log(Close_t)
    """
    y = np.log(df[price_column]).shift(-horizon) - np.log(df[price_column])
    y.name = f"future_logret_{horizon}"
    return y


def make_direction_label(y: pd.Series, threshold=0.0):
    """
    Convert continuous future returns → binary direction.
    """
    y_class = (y > threshold).astype(int)
    y_class.name = f"{y.name}_class"
    return y_class

class SupervisedLearning:

    def __init__(self):
        """Initialize all regression models."""
        self.models = []
        self.scaler = StandardScaler()
        self._init_models()
        self._init_regression()
        self._init_classifiers()

    # -----------------------------
    # Regression Models
    # -----------------------------
    def _init_regression(self):
        self.reg_models = [
            ("LR", LinearRegression()),
            ("LASSO", Lasso()),
            ("ElasticNet", ElasticNet()),
            ("DecisionTree", DecisionTreeRegressor()),
            ("RandomForest", RandomForestRegressor()),
            ("ExtraTrees", ExtraTreesRegressor()),
            ("GradientBoosting", GradientBoostingRegressor()),
            ("AdaBoost", AdaBoostRegressor()),
            ("SVR", SVR()),
            ("KNN", KNeighborsRegressor())
        ]

    # -----------------------------
    # Classification Models
    # -----------------------------
    def _init_classifiers(self):
        self.clf_models = [
            ("Logistic", LogisticRegression(max_iter=500)),
            ("RandomForest", RandomForestClassifier()),
            ("GradientBoosting", GradientBoostingClassifier()),
            ("KNN_Classifier", KNeighborsClassifier()),
        ]

    def get_classifier_by_name(self, name):
        for n, m in self.clf_models:
            if n == name:
                return m
        raise ValueError("Classifier name not found")


    def _init_models(self):
        # Linear Models
        self.models.append(("LR", LinearRegression()))
        self.models.append(("LASSO", Lasso()))
        self.models.append(("Elastic Net Penalty", ElasticNet()))

        # Tree-Based
        self.models.append(("Decision Tree", DecisionTreeRegressor()))

        # Bagging
        self.models.append(("Random Forest", RandomForestRegressor()))
        self.models.append(("Extra Trees", ExtraTreesRegressor()))

        # Boosting
        self.models.append(("Gradient Boosting", GradientBoostingRegressor()))
        self.models.append(("Adaptive Boosting", AdaBoostRegressor()))

        # Kernel / Distance-based
        self.models.append(("Support Vector Machine", SVR()))
        self.models.append(("K-Nearest Neighbors", KNeighborsRegressor()))

    def get_model_by_name(self, name):
        for n, model in self.models:
            if name == n:
                return model
            
        print("Warning! no name found")
        return self.models[0];


    def sequential_split(self, X: pd.DataFrame, Y: pd.Series, train_frac=0.8):
        if len(X) != len(Y):
            raise ValueError("X and Y must be equal length")

        if train_frac <= 0 or train_frac >= 1:
            raise ValueError("Train fraction must be between 0 and 1")

        train_size = int(len(X) * train_frac)

        X_train, X_test = X[0:train_size], X[train_size:]
        Y_train, Y_test = Y[0:train_size], Y[train_size:]

        print(f"Sequential Split: {len(X_train)} train / {len(X_test)} test samples")
        return X_train, X_test, Y_train, Y_test
    

    # Train ALL classifiers & report accuracy
    # -----------------------------
    def run_all_classifiers(self, X_train, Y_train, X_test, Y_test):
        results = []

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        for name, model in self.clf_models:
            model.fit(X_train_scaled, Y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(Y_test, y_pred)
            results.append((name, acc))
            print(f"{name}: accuracy={acc:.4f}")

        return pd.DataFrame(results, columns=["Model", "Accuracy"])

    def run_all_models(self, X_train, Y_train, X_test, Y_test, 
                       num_folds=10, seed=42, metric="neg_mean_squared_error"):
        
        # SCALE HERE
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        names, kfold_results, train_results, test_results = [], [], [], []

        for name, model in self.models:
            names.append(name)

            kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

            # Cross-validation (negative MSE -> convert to positive)
            cv_results = -1 * cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring=metric)
            kfold_results.append(cv_results)

            # Fit model
            res = model.fit(X_train_scaled, Y_train)

            # Compute train/test MSE
            train_mse = mean_squared_error(Y_train, res.predict(X_train_scaled))
            test_mse = mean_squared_error(Y_test, res.predict(X_test_scaled))

            train_results.append(train_mse)
            test_results.append(test_mse)

            # Display progress
            print(f"{name}: CV_Mean={cv_results.mean():.4f}, CV_Std={cv_results.std():.4f}, Train_MSE={train_mse:.4f}, Test_MSE={test_mse:.4f}")

        df_for_comparison = pd.DataFrame({
            "Algorithms": names * 2,
            "Data": ["Training Set"] * len(names) + ["Testing Set"] * len(names),
            "Performance": train_results + test_results
        })

        return {
            "names": names,
            "kfold_results": kfold_results,
            "train_results": train_results,
            "test_results": test_results,
            "comparison_df": df_for_comparison
        }

    def plot_performance(self, df_for_comparison: pd.DataFrame):
        performance_comparison =\
        (
            ggplot(df_for_comparison,
                aes(x = "Algorithms",
                    y = "Performance",
                    fill = "Data"
                    )
                )
            + geom_bar(stat = "identity",
                    position = "dodge",
                    width = 0.5)
            + labs(title = "Comparing the Performance of Machine Learning Algorithms on the Training vs. Testing Set",
                y = "Mean Squared Error (MSE)",
                x = "Name of ML Algorithms",
                caption = "Source: Federal Reserve Bank & Yahoo Finance")
            + theme(legend_position = "top")
            + ggsize(1000, 500)
        )

        performance_comparison.show()


    def tune_probability_threshold(
        self,
        model,
        X_train, y_train,
        X_test, y_test,
        thresholds = np.arange(0.50, 0.80, 0.01),
        return_metric = "sharpe"
    ):
        """
        Cross-validate classification thresholds.

        return_metric: 'sharpe' or 'accuracy'
        """

        if model is None:
            raise ValueError("Model cannot be None")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)


        # Fit classifier once
        model.fit(X_train_scaled, y_train)

        # Predicted probability of going UP
        p_train = model.predict_proba(X_train_scaled)[:, 1]
        p_test  = model.predict_proba(X_test_scaled)[:, 1]

        results = []

        for th in thresholds:

            train_pred = (p_train > th).astype(int)
            test_pred  = (p_test > th).astype(int)
            # Convert to +1/-1 returns strategy
            train_ret = train_pred * y_train
            test_ret  = test_pred * y_test

            # Compute Sharpe
            train_sharpe = (
                train_ret.mean() / train_ret.std() * np.sqrt(252)
                if train_ret.std() != 0 else np.nan
            )
            test_sharpe = (
                test_ret.mean() / test_ret.std() * np.sqrt(252)
                if test_ret.std() != 0 else np.nan
            )

            # Classification accuracy
            train_acc = accuracy_score(y_train, train_pred)
            test_acc  = accuracy_score(y_test, test_pred)

            results.append({
                "threshold": th,
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc
            })

        df = pd.DataFrame(results)

        # Pick the optimal threshold
        if return_metric == "sharpe":
            best_row = df.loc[df["train_sharpe"].idxmax()]
        else:
            best_row = df.loc[df["train_accuracy"].idxmax()]

        best_threshold = best_row["threshold"]

        print(f"\nBest Threshold = {best_threshold:.3f}")
        print(best_row)

        # Also compute final predictions on test set
        final_test_pred = (p_test > best_threshold).astype(int)

        return {
            "threshold_cv_results": df,
            "best_threshold": best_threshold,
            "best_row": best_row,
            "final_test_pred": final_test_pred,
            "model": model
        }