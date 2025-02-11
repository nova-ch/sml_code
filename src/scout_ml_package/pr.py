import time
import pandas as pd
from scout_ml_package.utils.demo import DummyData, DataValidator, FakeListener
from scout_ml_package.model.model_pipeline import (
    ModelManager,
    PredictionPipeline,
)

from scout_ml_package.utils.logger import configure_logger
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
#logger = configure_logger('prediction_logger', '/data/model-data/logs/prediction_logger.log')
logger = configure_logger('demo_logger', '/data/model-data/logs', 'pred.log')

# Define acceptable ranges for each prediction
acceptable_ranges = {
    # Adjust these ranges based on your domain knowledge
    "RAMCOUNT": (100, 10000),
    "CTIME": (0.1, 10000),
    "CPU_EFF": (0, 100),
}

additional_ctime_ranges = {
    "low": (0.1, 10),
    "high": (99, 10000),
}

# def get_prediction(model_manager, r):
#     start_time = time.time()

#     try:
#         if r is not None:
#             # Check if DataFrame is not empty before accessing elements
#             if not r.empty:
#                 jeditaskid = r["JEDITASKID"].values[0]
#                 processor = PredictionPipeline(model_manager)
#                 base_df = processor.preprocess_data(r)

#                 # Model 1: RAMCOUNT
#                 features = (
#                     ["JEDITASKID"]
#                     + processor.numerical_features
#                     + processor.category_sequence
#                 )
#                 base_df.loc[:, "RAMCOUNT"] = (
#                     processor.make_predictions_for_model(
#                         "1", features, base_df
#                     )
#                 )
#                 DataValidator.validate_prediction(base_df, "RAMCOUNT", acceptable_ranges, jeditaskid)

#                 # Model 2 and 3: cputime_HS
#                 processor.numerical_features.append("RAMCOUNT")
#                 features = (
#                     ["JEDITASKID"]
#                     + processor.numerical_features
#                     + processor.category_sequence
#                 )

#                 if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
#                     base_df.loc[:, "CTIME"] = (
#                         processor.make_predictions_for_model(
#                             "2", features, base_df
#                         )
#                     )
#                 else:
#                     base_df.loc[:, "CTIME"] = (
#                         processor.make_predictions_for_model(
#                             "3", features, base_df
#                         )
#                     )
#                 DataValidator.validate_ctime_prediction(base_df, jeditaskid, acceptable_ranges, additional_ctime_ranges)

#                 # Model 4: CPU_EFF
#                 processor.numerical_features.append("CTIME")
#                 features = (
#                     ["JEDITASKID"]
#                     + processor.numerical_features
#                     + processor.category_sequence
#                 )
#                 base_df.loc[:, "CPU_EFF"] = (
#                     processor.make_predictions_for_model(
#                         "4", features, base_df
#                     )
#                 )
#                 DataValidator.validate_prediction(base_df, "CPU_EFF", acceptable_ranges, jeditaskid)

#                 # Model 5: IOINTENSITY
#                 processor.numerical_features.append("CPU_EFF")
#                 features = (
#                     ["JEDITASKID"]
#                     + processor.numerical_features
#                     + processor.category_sequence
#                 )
#                 base_df.loc[:, "IOINTENSITY"] = (
#                     processor.make_predictions_for_model(
#                         "5", features, base_df
#                     )
#                 )

#                 logger.info(
#                     f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
#                 )
#                 base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']] = base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']].round(3)
#                 return base_df

#             else:
#                 logger.error("DataFrame is empty for JEDITASKID.")
#                 return None

#         else:
#             logger.error("Failed to process: Input data is None")
#             return None

#     except IndexError as ie:
#         logger.error(f"IndexError occurred: {ie}")
#         return None

#     except ValueError as ve:
#         jeditaskid = (
#             "Unknown" if r is None or r.empty else r["JEDITASKID"].values[0]
#         )
#         logger.error(f"Check failed for JEDITASKID {jeditaskid}: {ve}")
#         return None

#     except Exception as e:
#         jeditaskid = (
#             "Unknown" if r is None or r.empty else r["JEDITASKID"].values[0]
#         )
#         logger.error(f"Error processing JEDITASKID {jeditaskid}: {str(e)}")
#         return None

def get_prediction(model_manager, r):
    start_time = time.time()

    try:
        if r is None or r.empty:
            logger.error("Input data is None or empty.")
            return None

        jeditaskid = r["JEDITASKID"].values[0]
        processor = PredictionPipeline(model_manager)
        base_df = processor.preprocess_data(r)

        # Model 1: RAMCOUNT
        features = (
            ["JEDITASKID"]
            + processor.numerical_features
            + processor.category_sequence
        )
        try:
            base_df.loc[:, "RAMCOUNT"] = (
                processor.make_predictions_for_model(
                    "1", features, base_df
                )
            )
            DataValidator.validate_prediction(base_df, "RAMCOUNT", acceptable_ranges, jeditaskid)
        except Exception as e:
            logger.error(f"Error processing RAMCOUNT for JEDITASKID {jeditaskid}: {str(e)}")
            return None

        # Model 2 and 3: cputime_HS
        processor.numerical_features.append("RAMCOUNT")
        features = (
            ["JEDITASKID"]
            + processor.numerical_features
            + processor.category_sequence
        )
        try:
            if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
                base_df.loc[:, "CTIME"] = (
                    processor.make_predictions_for_model(
                        "2", features, base_df
                    )
                )
            else:
                base_df.loc[:, "CTIME"] = (
                    processor.make_predictions_for_model(
                        "3", features, base_df
                    )
                )
            DataValidator.validate_ctime_prediction(base_df, jeditaskid, acceptable_ranges, additional_ctime_ranges)
        except Exception as e:
            logger.error(f"Error processing CTIME for JEDITASKID {jeditaskid}: {str(e)}")
            return None  # Return immediately if CTIME fails

        # Only proceed if CTIME was successful
        # Model 4: CPU_EFF
        processor.numerical_features.append("CTIME")
        features = (
            ["JEDITASKID"]
            + processor.numerical_features
            + processor.category_sequence
        )
        try:
            base_df.loc[:, "CPU_EFF"] = (
                processor.make_predictions_for_model(
                    "4", features, base_df
                )
            )
            DataValidator.validate_prediction(base_df, "CPU_EFF", acceptable_ranges, jeditaskid)
        except Exception as e:
            logger.error(f"Error processing CPU_EFF for JEDITASKID {jeditaskid}: {str(e)}")
            return None

        # Model 5: IOINTENSITY
        processor.numerical_features.append("CPU_EFF")
        features = (
            ["JEDITASKID"]
            + processor.numerical_features
            + processor.category_sequence
        )
        try:
            base_df.loc[:, "IOINTENSITY"] = (
                processor.make_predictions_for_model(
                    "5", features, base_df
                )
            )
        except Exception as e:
            logger.error(f"Error processing IOINTENSITY for JEDITASKID {jeditaskid}: {str(e)}")
            return None

        logger.info(
            f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
        )
        base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']] = base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']].round(3)
        return base_df

    except Exception as e:
        jeditaskid = (
            "Unknown" if r is None or r.empty else r["JEDITASKID"].values[0]
        )
        logger.error(f"Unexpected error processing JEDITASKID {jeditaskid}: {str(e)}")
        return None



if __name__ == "__main__":
    df = DummyData.fetch_data()
    # base_path = "/Users/tasnuvachowdhury/Desktop/PROD/pandaml-test/src/"
    base_path = "/data/model-data/"  # "/data/test/"
    model_manager = ModelManager(base_path)
    model_manager.load_models()

    #db_fetcher = DatabaseFetcher()
    input_db = DatabaseFetcher('database')
    output_db = DatabaseFetcher('output_database')
    
    cols_to_write = ['JEDITASKID', 'PRODSOURCELABEL', 'PROCESSINGTYPE', 'TRANSHOME',
       'CPUTIMEUNIT', 'CORECOUNT', 'TOTAL_NFILES', 'TOTAL_NEVENTS',
       'DISTINCT_DATASETNAME_COUNT', 'RAMCOUNT', 'CTIME', 'CPU_EFF',
       'IOINTENSITY']
    query = """
    SELECT * FROM ATLAS_PANDA.PANDAMLTEST
    """
    test = pd.read_sql(query, con=output_db.get_connection())
    print(test)
    print(test.columns)
    sample_tasks = [27766704, 27746332, 30749131, 30752901]
    listener = FakeListener(sample_tasks, delay=6)  # Pass delay here
    for (
        jeditaskid
    ) in listener.demo_task_listener():  # No arguments needed here
        print(f"Received JEDITASKID: {jeditaskid}")
        r = input_db.fetch_task_param(jeditaskid)
        # r = df[df['JEDITASKID'] == jeditaskid].copy()
        print(r)
        result = get_prediction(model_manager, r)
     
        if result is not None:
            logger.info("Processing completed successfully")
            print(result.columns)
            result = result[cols_to_write]
            output_db.write_data(result, 'ATLAS_PANDA.PANDAMLTEST')
        else:
            logger.error("Processing failed due to invalid results or errors")
        print("Next Trial")
        print(result)

        # # Add a 10-minute delay here
        # print("Waiting for 10 minutes before processing the next task...")
        # time.sleep(4)  # Reduced delay for testing
        # # "Wake up" actions
        # print("Waking up after 10 minutes sleep")
        # logger.info("Resuming execution after sleep period")
        # # You can add any other actions you want to perform after waking up here

    print("All tasks processed")
    input_db.close_connection()

