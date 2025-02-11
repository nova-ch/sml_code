import time
import pandas as pd
#import logging

from scout_ml_package.utils.logger import configure_logger
#logger = configure_logger('demo_logger', '/data/model-data/logs/')

logger = configure_logger('demo_logger', '/data/model-data/logs', 'demo.log', log_level=logging.ERROR)

class FakeListener:
    def __init__(self, task_ids, delay=3):
        self.task_ids = task_ids
        self.delay = delay

    def demo_task_listener(self):
        """
        A demo function that simulates a listener sending task IDs with a fixed delay.

        Yields:
        int: A task ID from the list.
        """
        for task_id in self.task_ids:
            # Wait for the specified delay
            time.sleep(self.delay)
            yield task_id


# class DataValidator:
#     @classmethod
#     def check_predictions(cls, df, column, acceptable_ranges):
#         min_val, max_val = acceptable_ranges[column]
#         if df[column].min() < min_val or df[column].max() > max_val:
#             raise ValueError(
#                 f"Predictions for {column} are outside the acceptable range of {acceptable_ranges[column]}"
#             )

class DataValidator:
    @classmethod
    def check_predictions(cls, df, column, acceptable_ranges):
        min_val, max_val = acceptable_ranges[column]
        if df[column].min() < min_val or df[column].max() > max_val:
            raise ValueError(
                f"Predictions for {column} are outside the acceptable range of {acceptable_ranges[column]}"
            )
        return True

    @classmethod
    def validate_prediction(cls, df, column, acceptable_ranges, jeditaskid):
        """
        Validates predictions for a given column and logs the result.

        Parameters:
        - df: DataFrame containing predictions.
        - column: Column name to validate.
        - acceptable_ranges: Acceptable ranges for validation.
        - jeditaskid: ID for logging purposes.

        Returns:
        - bool: True if validation succeeds, False otherwise.
        """
        try:
            cls.check_predictions(df, column, acceptable_ranges)
            logger.info(f"{column} predictions validated successfully.")
            return True
        except ValueError as ve:
            logger.error(f"{column} validation failed for JEDITASKID {jeditaskid}: {ve}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during {column} validation for JEDITASKID {jeditaskid}: {e}")
            return False
            
    @classmethod
    def validate_ctime_prediction(cls, df, jeditaskid, additional_ctime_ranges):
        """
        Validates CTIME predictions using alternative ranges.
    
        Parameters:
        - df: DataFrame containing predictions.
        - jeditaskid: ID for logging purposes.
        - additional_ctime_ranges: Alternative ranges for CTIME validation.
    
        Returns:
        - bool: True if validation succeeds, False otherwise.
        """
        try:
            if df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
                cls.check_predictions(df, "CTIME", {"CTIME": additional_ctime_ranges["low"]})
                logger.info("Validation passed with low CTIME range.")
                return True
            else:
                cls.check_predictions(df, "CTIME", {"CTIME": additional_ctime_ranges["high"]})
                logger.info("Validation passed with high CTIME range.")
                return True
        except ValueError as ve:
            logger.error(f"Validation failed with all ranges: {ve}")
            return False

    # @classmethod
    # def validate_ctime_prediction(cls, df, jeditaskid, acceptable_ranges, additional_ctime_ranges):
    #     """
    #     Validates CTIME predictions with alternative ranges if necessary.

    #     Parameters:
    #     - df: DataFrame containing predictions.
    #     - jeditaskid: ID for logging purposes.
    #     - acceptable_ranges: Default acceptable ranges for validation.
    #     - additional_ctime_ranges: Alternative ranges for CTIME validation.

    #     Returns:
    #     - bool: True if validation succeeds, False otherwise.
    #     """
    #     try:
    #         cls.check_predictions(df, "CTIME", acceptable_ranges)
    #         logger.info("CTIME predictions passed validation with default range.")
    #         return True
    #     except ValueError as ve:
    #         logger.warning(f"Validation failed with default range: {ve}")
            
    #         # Attempt alternative ranges
    #         try:
    #             if df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
    #                 cls.check_predictions(df, "CTIME", {"CTIME": additional_ctime_ranges["low"]})
    #                 logger.info("Validation passed with low CTIME range.")
    #                 return True
    #             else:
    #                 cls.check_predictions(df, "CTIME", {"CTIME": additional_ctime_ranges["high"]})
    #                 logger.info("Validation passed with high CTIME range.")
    #                 return True
    #         except ValueError as ve_alt:
    #             logger.error(f"Validation failed with all ranges: {ve_alt}")
    #             return False






class DummyData:
    @classmethod
    def fetch_data(cls):
        # Simulate data retrieval
        data = {
            "JEDITASKID": [27766704, 27746332],
            "PRODSOURCELABEL": ["managed", "user"],
            "PROCESSINGTYPE": ["deriv", "panda-client-1.4.98-jedi-run"],
            "TRANSHOME": [
                "AthDerivation-21.2.77.0",
                "AnalysisTransforms-AnalysisBase_21.2.197",
            ],
            "CPUTIMEUNIT": ["HS06sPerEvent", "mHS06sPerEvent"],
            "CORECOUNT": [8, 1],
            "TOTAL_NFILES": [290000, 11237955],
            "TOTAL_NEVENTS": [23, 260],
            "DISTINCT_DATASETNAME_COUNT": [1, 3],
        }
        return pd.DataFrame(data)


#
# # Example usage
# if __name__ == "__main__":
#     task_ids = [27766704, 27746332]
#     task_manager = TaskManager(task_ids, delay=3)
#
#     data_fetcher = DataFetcher()
#     df = data_fetcher.fetch_data()
#
#     acceptable_ranges = {
#         'RAMCOUNT': (100, 10000),
#         'cputime_HS': (0.4, 10000),
#         'CPU_EFF': (0, 100)
#     }
#     data_validator = DataValidator(acceptable_ranges)
#
#     for jeditaskid in task_manager.demo_task_listener():
#         print(f"Received JEDITASKID: {jeditaskid}")
#         r = df[df['JEDITASKID'] == jeditaskid].copy()
#         print(r)
#
#         # Example of using data_validator
#         # data_validator.check_predictions(r, 'RAMCOUNT')
#
#         print("Next Trial")
#         print("Waiting for 10 minutes before processing the next task...")
#         time.sleep(4)  # Reduced delay for testing
#         print("Waking up after 10 minutes sleep")
#
#     print("All tasks processed")
