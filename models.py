import catboost as cb
import lightgbm as lgbm
import numpy as np
import optuna
import tensorflow as tf
import xgboost as xgb
import yaml
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# read nested config.yaml file into a python dictionary
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def smape(y_true, y_pred):
    """
    Calculate SMAPE score

    Args:
        y_true (pd.DataFrame): The true target
        y_pred (pd.DataFrame): The predicted target
    """
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def apply_random_forest(X, y, X_val, y_val):
    """
    Apply Random Forest to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
        X_val (pd.DataFrame): The validation features
        y_val (pd.DataFrame): The validation target

    Returns:
        rf (sklearn.ensemble.RandomForestRegressor): The trained model
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical(
            "max_features", [1.0, "sqrt", "log2"])
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
        )
        rf.fit(X, y)
        # calculate SMAPE score
        y_pred = rf.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1, show_progress_bar=True)
    rf = RandomForestRegressor(**study.best_params)
    rf.fit(X, y)
    return rf


def apply_xgboost(X, y, X_val, y_val):
    """
    Apply XGBoost to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target

    Returns:
        xgb (xgboost.XGBRegressor): The trained model
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        xgb = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
        xgb.fit(X, y)
        # calculate SMAPE score
        y_pred = xgb.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    xgb = xgb.XGBRegressor(**study.best_params)
    xgb.fit(X, y)
    return xgb


def apply_lightgbm(X, y, X_val, y_val):
    """
    Apply LightGBM to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
        X_val (pd.DataFrame): The validation features
        y_val (pd.DataFrame): The validation target

    Returns:
        lgbm (lightgbm.LGBMRegressor): The trained model
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 1, 10)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        lgbm = lgbm.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
        lgbm.fit(X, y)
        # calculate SMAPE score
        y_pred = lgbm.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    lgbm = lgbm.LGBMRegressor(**study.best_params)
    lgbm.fit(X, y)
    return lgbm


def apply_catboost(X, y, X_val, y_val):
    """
    Apply CatBoost to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
        X_val (pd.DataFrame): The validation features
        y_val (pd.DataFrame): The validation target

    Returns:
        cat (catboost.CatBoostRegressor): The trained model
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 1, 10)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        cat = cat.CatBoostRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
        cat.fit(X, y)
        # calculate SMAPE score
        y_pred = cat.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    cat = cat.CatBoostRegressor(**study.best_params)
    cat.fit(X, y)
    return cat


def apply_decision_tree(X, y, X_val, y_val):
    """
    Apply Decision Tree to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
        X_val (pd.DataFrame): The validation features
        y_val (pd.DataFrame): The validation target

    Returns:
        dt (sklearn.tree.DecisionTreeRegressor): The trained model
    """
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        dt = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        dt.fit(X, y)
        # calculate SMAPE score
        y_pred = dt.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    dt = DecisionTreeRegressor(**study.best_params)
    dt.fit(X, y)
    return dt


def apply_gradient_boosting(X, y, X_val, y_val):
    """
    Apply gradient boosting to the data and apply hyperparameter tuning using Optuna

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
        X_val (pd.DataFrame): The validation features
        y_val (pd.DataFrame): The validation target

    Returns:
        gb (sklearn.ensemble.GradientBoostingRegressor): The trained model
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        gb.fit(X, y)
        # calculate SMAPE score
        y_pred = gb.predict(X_val)
        return smape(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    gb = GradientBoostingRegressor(**study.best_params)
    gb.fit(X, y)
    return gb


def apply_knn(X, y):
    """
    Apply KNN to the data

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
    """
    knn = KNeighborsRegressor(**config["knn_params"])
    knn.fit(X, y)
    return knn


def apply_adaboost(X, y):
    """
    Apply AdaBoost to the data

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
    """
    ada = AdaBoostRegressor(**config["ada_params"])
    ada.fit(X, y)
    return ada


def apply_svm(X, y):
    """
    Apply SVM to the data

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
    """
    svm = SVR(**config["svm_params"])
    svm.fit(X, y)
    return svm


def apply_neural_network(X, y):
    """
    Apply Neural Network to the data

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
    """
    # Fix: ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
    X = X.astype(float)
    y = y.astype(float)

    # Initialize the model
    nn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    nn.compile(loss="mse", optimizer="adam")
    nn.fit(X, y, epochs=10, verbose=0)
    return nn


def apply_lstm(X, y):
    """
    Apply LSTM to the data

    Args:
        X (pd.DataFrame): The features
        y (pd.DataFrame): The target
    """
    X = X.astype(float)
    y = y.astype(float)
    lstm = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    lstm.compile(loss="mse", optimizer="adam")
    lstm.fit(X, y, epochs=10, verbose=0)
    return lstm
