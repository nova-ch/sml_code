import time
import pandas as pd


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


class DataValidator:
    @classmethod
    def check_predictions(cls, df, column, acceptable_ranges):
        min_val, max_val = acceptable_ranges[column]
        if df[column].min() < min_val or df[column].max() > max_val:
            raise ValueError(
                f"Predictions for {column} are outside the acceptable range of {acceptable_ranges[column]}"
            )


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
