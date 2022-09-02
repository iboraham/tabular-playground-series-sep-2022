import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import argparse

import fe
import models
import logging
import helpers


def get_data() -> pd.DataFrame:
    """
    Get the data from the csv file

    Returns:
        pd.DataFrame: The data -- columns = [row_id, date,country, store, product, num_sold]
    """
    return pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")


def main(args):
    """
    Predict the number of products sold in the future
    """
    logging.basicConfig(level=logging.INFO)

    data, test = get_data()
    if args.verbose:
        logging.info("Data loaded!")

    data, test = fe.FeatureEngineer(data, test).fit()

    if args.verbose:
        logging.info("Data feature engineered!")

    # TODO: split data to train and validation in advanced way
    train, val = train_test_split(data, test_size=0.2, random_state=42)
    X, y = train.drop("num_sold", axis=1), train["num_sold"]
    X_val, y_val = val.drop("num_sold", axis=1), val["num_sold"]

    # Apply random forest to the data
    rf = models.apply_random_forest(X, y, X_val, y_val)
    if args.verbose:
        logging.info("Random forest applied!")
        logging.info(
            f"Random forest R^2 score: {r2_score(y_val, rf.predict(X_val))}")

    # Apply XGBoost to the data
    xgb = models.apply_xgboost(X, y)
    if args.verbose:
        logging.info("XGBoost applied!")
        logging.info(
            f"XGBoost R^2 score: {r2_score(y_val, xgb.predict(X_val))}")

    # # Apply LightGBM to the data
    lgb = models.apply_lightgbm(X, y)
    if args.verbose:
        logging.info("LightGBM applied!")
        logging.info(
            f"LightGBM R^2 score: {r2_score(y_val, lgb.predict(X_val))}")

    # Apply CatBoost to the data
    cat = models.apply_catboost(X, y)
    if args.verbose:
        logging.info("CatBoost applied!")
        logging.info(
            f"CatBoost R^2 score: {r2_score(y_val, cat.predict(X_val))}")

    # Apply Neural Network to the data
    # nn = models.apply_neural_network(X, y)
    # if args.verbose:
    #     logging.info("Neural Network applied!")
    #     logging.info(
    #         f"Neural Network R^2 score: {r2_score(y_val, nn.predict(X_val.astype(float)))}")

    # lstm = models.apply_lstm(X, y)
    # if args.verbose:
    #     logging.info("LSTM applied!")
    #     logging.info(
    #         f"LSTM R^2 score: {r2_score(y_val, lstm.predict(X_val.astype(float)))}")

    # # Apply Decision Tree to the data
    dt = models.apply_decision_tree(X, y)
    if args.verbose:
        logging.info("Decision Tree applied!")
        logging.info(
            f"Decision Tree R^2 score: {r2_score(y_val, dt.predict(X_val))}")

    # Apply Gradient Boosting to the data
    gb = models.apply_gradient_boosting(X, y)
    if args.verbose:
        logging.info("Gradient Boosting applied!")
        logging.info(
            f"Gradient Boosting R^2 score: {r2_score(y_val, gb.predict(X_val))}")

    # Ensemble the models and make predictions
    if args.model == "all":
        helpers.ensemble_models(X_val=X_val, y_val=y_val, data=test,
                                rf=rf, xgb=xgb, lgb=lgb, cat=cat, dt=dt, gb=gb
                                )
    else:
        helpers.predict(data=test, model=eval(
            args.model), model_name=args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()
    assert args.model in ["rf", "xgb", "lgb", "cat", "dt", "ada", "gb", "all"]
    main(args)
