import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelMetrics:
    def __init__(self, actual_columns: list):
        """
        Initializes the ModelMetrics class.

        Parameters:
        actual_columns (list): List of the names of the columns containing actual target values in the DataFrame.
        """
        self.actual_columns = actual_columns

    def calculate_metrics(self, df: pd.DataFrame, predicted_columns: list) -> dict:
        """
        Calculates error metrics for predicted values compared to actual values.

        Parameters:
        df (pd.DataFrame): DataFrame containing actual and predicted values.
        predicted_columns (list): List of column names that contain predicted values.

        Returns:
        dict: A dictionary containing the error metrics for each model and overall metrics.
        """
        results = {}
        overall_mae = 0
        overall_rmse = 0
        overall_mape = 0
        overall_r2 = 0
        num_models = len(predicted_columns)

        for predicted_column in predicted_columns:
            # Initialize accumulators for this model
            model_mae = 0
            model_rmse = 0
            model_mape = 0
            model_r2 = 0

            for actual_column in self.actual_columns:
                # Extract actual and predicted values
                actual = df[actual_column]
                predicted = df[predicted_column]

                # Calculate error metrics
                mae = mean_absolute_error(actual, predicted)
                rmse = mean_squared_error(actual, predicted, squared=False)
                r2 = r2_score(actual, predicted)

                # Calculate Mean Absolute Percentage Error
                mape = ( abs(actual - predicted) / actual).mean() * 100 if not actual.empty and actual.mean() != 0 else float(
                    'inf')  # Handle potential division by zero

                # Accumulate model metrics
                model_mae += mae
                model_rmse += rmse
                model_mape += mape
                model_r2 += r2

            # Average metrics for current model
            num_actuals = len(self.actual_columns)
            results[predicted_column] = {
                'Mean Absolute Error': model_mae / num_actuals,
                'Root Mean Squared Error': model_rmse / num_actuals,
                'R-squared': model_r2 / num_actuals,
                'Mean Absolute Percentage Error': model_mape / num_actuals,
            }

            # Accumulate overall metrics
            overall_mae += model_mae / num_actuals
            overall_rmse += model_rmse / num_actuals
            overall_mape += model_mape / num_actuals
            overall_r2 += model_r2 / num_actuals

        # Calculate overall metrics (averaging)
        overall_metrics = {
            'Overall Mean Absolute Error': overall_mae / num_models if num_models > 0 else 0,
            'Overall Root Mean Squared Error': overall_rmse / num_models if num_models > 0 else 0,
            'Overall Mean Absolute Percentage Error': overall_mape / num_models if num_models > 0 else 0,
            'Overall R-squared': overall_r2 / num_models if num_models > 0 else 0,
        }

        return {
            'Individual Metrics': results,
            'Overall Metrics': overall_metrics
        }