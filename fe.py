import pandas as pd
import holidays
import pycountry


class FeatureEngineer:
    """
    Feature Engineer the data
    """

    def __init__(self, data: pd.DataFrame, test: pd.DataFrame):
        """
        Initialize the FeatureEngineer class

        Args:
            data (pd.DataFrame): The data
        """
        self.data = data
        self.test = test

    def fit(self) -> pd.DataFrame:
        """
        Perform feature engineering on the data

        Returns:
            pd.DataFrame: The data with new features
        """
        # Perform feature engineering on the data
        self.data = feature_engineering_date(self.data)
        self.test = feature_engineering_date(self.test)

        self.data, self.test = feature_engineering_product(
            self.data, self.test)
        self.data, self.test = feature_engineering_country(
            self.data, self.test)
        self.data, self.test = feature_engineering_store(self.data, self.test)
        return self.data, self.test


def _handle_categorical(data, test, col):
    # Extract the product features
    data[col] = data[col].astype("category")

    # Apply same encoding to test data
    test[col] = test[col].astype("category")
    test[col] = test[col].cat.set_categories(
        data[col].cat.categories)
    assert all(data[col].cat.categories) == all(
        test[col].cat.categories)

    data[col] = data[col].cat.codes
    test[col] = test[col].cat.codes
    return data, test


def feature_engineering_product(data: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the data by extracting the product features

    Args:
        data (pd.DataFrame): The data

    Returns:
        pd.DataFrame: The data with new features
    """
    return _handle_categorical(data, test, col='product')


def feature_engineering_country(data: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the data by extracting the country features

    Args:
        data (pd.DataFrame): The data

    Returns:
        pd.DataFrame: The data with new features
    """
    return _handle_categorical(data, test, col='country')


def feature_engineering_store(data: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the data by extracting the store features

    Args:
        data (pd.DataFrame): The data

    Returns:
        pd.DataFrame: The data with new features
    """
    # Extract the store features
    return _handle_categorical(data, test, col='store')


def feature_engineering_date(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the data by extracting the date features

    Args:
        data (pd.DataFrame): The data

    Returns:
        pd.DataFrame: The data with new features
    """
    # Convert date to datetime
    data["date"] = pd.to_datetime(data["date"])

    # Extract year, month, day, day of week, and day of year from date
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    # Add seasonality features
    data["is_winter"] = data["month"].isin([12, 1, 2])
    data["is_spring"] = data["month"].isin([3, 4, 5])
    data["is_summer"] = data["month"].isin([6, 7, 8])
    data["is_autumn"] = data["month"].isin([9, 10, 11])

    # Add weekend feature
    data["is_weekend"] = data["day_of_week"].isin([5, 6])

    # Add holiday feature
    data["is_holiday"] = data.apply(
        lambda x: x['date'] in holidays.country_holidays(pycountry.countries.get(name=x['country']).alpha_2), axis=1)

    # TODO: Add more features

    # Drop date
    data = data.drop("date", axis=1)

    return data
