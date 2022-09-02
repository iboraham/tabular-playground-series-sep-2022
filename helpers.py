import logging

import pandas as pd
from sklearn.metrics import r2_score
from models import smape


def ensemble_models(X_val, y_val, data, **kwargs):
    """
    Calculate each model's predictions and ensemble them using weighted average where weights are the model's smape scores

    Args:
        data (pd.DataFrame): The data
        **kwargs: The models
    """

    # Calculate each model's predictions
    predictions = {
        model_name: model.predict(X_val) for model_name, model in kwargs.items()
    }
    predictions_data = {
        model_name: model.predict(data) for model_name, model in kwargs.items()
    }

    # Calculate each model's SMAPE score
    smape_scores = {
        model_name: smape(y_val, predictions[model_name])
        for model_name in predictions
    }

    # Calculate the weights
    weights = {
        model_name: (1/smape_scores[model_name]) / sum(smape_scores.values())
        for model_name in smape_scores
    }

    # Calculate the weighted average
    ensemble_val = sum(
        [predictions[model_name] * weights[model_name]
            for model_name in predictions]
    )
    # Calculate SMAPE score of the ensemble
    logging.info(
        f"Ensemble SMAPE score: {smape(y_val, ensemble_val)}")

    ensemble = sum(
        [predictions_data[model_name] * weights[model_name]
            for model_name in predictions_data]
    )

    # Save the predictions to a csv file
    pd.DataFrame(
        {"row_id": data["row_id"], "num_sold": ensemble.astype(int)}).to_csv("predictions_ensemble.csv", index=False)


def predict(model, data, model_name=""):
    """
    Predict the number of products sold in the future

    Args:
        model (sklearn model): The model
        data (pd.DataFrame): The data
    """
    # Predict the number of products sold in the future
    predictions = model.predict(data)

    # Save the predictions to a csv file
    pd.DataFrame(
        {"row_id": data["row_id"], "num_sold": predictions.astype(int)}).to_csv(f"predictions{model_name}.csv", index=False)
