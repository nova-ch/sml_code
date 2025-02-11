# src/scout_ml_package/prediction_pipeline.py
import time
import logging
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from scout_ml_package.utils.demo import DummyData, DataValidator, FakeListener
from scout_ml_package.model.model_pipeline import (
    ModelManager,
    PredictionPipeline,
)
from scout_ml_package.data.fetch_db_data import DatabaseFetcher

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add a StreamHandler to print logs to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Add a FileHandler to write logs to a file
log_path = "/data/model-data/logs/prediction_pipeline.log"
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


logger.info("Logging test: This should appear in both console and file.")

# Define acceptable ranges for each prediction
acceptable_ranges = {
    # Adjust these ranges based on your domain knowledge
    "RAMCOUNT": (100, 10000),
    "CTIME": (0.1, 10000),
    "CPU_EFF": (0, 100),
}

additional_ctime_ranges = {
    "low": (0.1, 10),
    "high": (400, 10000),
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
#                 DataValidator.check_predictions(
#                     base_df, "RAMCOUNT", acceptable_ranges
#                 )

#                 try:
#                     DataValidator.check_predictions(base_df, "RAMCOUNT", acceptable_ranges)
#                     logger.info("RAMCOUNT predictions validated successfully.")
#                 except ValueError as ve:
#                     logger.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}: {ve}")
#                 except Exception as e:
#                     logger.error(f"Unexpected error during RAMCOUNT validation for JEDITASKID {jeditaskid}: {e}")

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
#                 DataValidator.check_predictions(
#                     base_df, "CTIME", acceptable_ranges
#                 )

#                 try:
#                     DataValidator.check_predictions(base_df, "CTIME", acceptable_ranges)
#                     logger.info("CTIME predictions passed validation with default range.")
#                 except ValueError as ve:
#                     logger.warning(f"Validation failed with default range: {ve}")
                    
#                     # Attempt alternative ranges
#                     try:
#                         if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
#                             DataValidator.check_predictions(base_df, "CTIME", {"CTIME": additional_ctime_ranges["low"]})
#                             logger.info("Validation passed with low CTIME range.")
#                         else:
#                             DataValidator.check_predictions(base_df, "CTIME", {"CTIME": additional_ctime_ranges["high"]})
#                             logger.info("Validation passed with high CTIME range.")
#                     except ValueError as ve_alt:
#                         logger.error(f"Validation failed with all ranges: {ve_alt}")
                

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
#                 DataValidator.check_predictions(
#                     base_df, "CPU_EFF", acceptable_ranges
#                 )

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

#                 logging.info(
#                     f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
#                 )
#                 base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']] = base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']].apply(lambda x: np.around(x, decimals=3))
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
    
    if r is None or r.empty:
        logger.error("DataFrame is empty or input data is None.")
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
    base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model("1", features, base_df)
    
    if not DataValidator.validate_prediction(base_df, "RAMCOUNT", acceptable_ranges, jeditaskid):
        logger.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}.")
        return f"M1 failure."

    # Update features for subsequent models
    processor.numerical_features.append("RAMCOUNT")
    features = (
        ["JEDITASKID"]
        + processor.numerical_features
        + processor.category_sequence
    )

    # Model 2/3: CTIME
    try:
        if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
            base_df.loc[:, "CTIME"] = processor.make_predictions_for_model("2", features, base_df)
        else:
            base_df.loc[:, "CTIME"] = processor.make_predictions_for_model("3", features, base_df)
        
        if not DataValidator.validate_ctime_prediction(base_df, jeditaskid, additional_ctime_ranges):
            logger.error(f"CTIME validation failed for JEDITASKID {jeditaskid}.")
            cpu_unit = base_df["CPUTIMEUNIT"].values[0]
            if cpu_unit == "mHS06sPerEvent":
                return f"M2 failure"
            else:
                return f"M3 failure"
    except Exception as e:
        logger.error(f"CTIME prediction failed for JEDITASKID {jeditaskid}: {str(e)}")
        cpu_unit = base_df["CPUTIMEUNIT"].values[0]
        if cpu_unit == "mHS06sPerEvent":
            return f"{jeditaskid}M2 failure: {str(e)}"
        else:
            return f"{jeditaskid}M3 failure: {str(e)}"

    # Update features for subsequent models
    processor.numerical_features.append("CTIME")
    features = (
        ["JEDITASKID"]
        + processor.numerical_features
        + processor.category_sequence
    )

    # Model 4: CPU_EFF
    try:
        base_df.loc[:, "CPU_EFF"] = processor.make_predictions_for_model("4", features, base_df)
        if not DataValidator.validate_prediction(base_df, "CPU_EFF", acceptable_ranges, jeditaskid):
            logger.error(f"CPU_EFF validation failed for JEDITASKID {jeditaskid}.")
            return f"{jeditaskid}M4 failure: Validation failed."
    except Exception as e:
        logger.error(f"{jeditaskid} M4 failure: {str(e)}")
        return f"M4 failure: {str(e)}"

    # Update features for subsequent models
    processor.numerical_features.append("CPU_EFF")
    features = (
        ["JEDITASKID"]
        + processor.numerical_features
        + processor.category_sequence
    )

    # Model 5: IOINTENSITY
    try:
        base_df.loc[:, "IOINTENSITY"] = processor.make_predictions_for_model("5", features, base_df)
    except Exception as e:
        logger.error(f"{jeditaskid}M5 failure: {str(e)}")
        return f"M5 failure: {str(e)}"

    logger.info(
        f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
    )
    base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']] = base_df[['RAMCOUNT', 'CTIME', 'CPU_EFF']].round(3)
    return base_df






import pandas as pd
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
    sample_tasks = [30752901] #[27766704, 27746332, 30749131, 30752901]
    listener = FakeListener(sample_tasks, delay=6)  # Pass delay here
    for (
        jeditaskid
    ) in listener.demo_task_listener():  # No arguments needed here
        print(f"Received JEDITASKID: {jeditaskid}")
        r = input_db.fetch_task_param(jeditaskid)
        # r = df[df['JEDITASKID'] == jeditaskid].copy()
        print(r)
        result = get_prediction(model_manager, r)
        print(result)
     
        if isinstance(result, pd.DataFrame):
            logging.info("Processing completed successfully")
            print(result.columns)
            result = result[cols_to_write]
            output_db.write_data(result, 'ATLAS_PANDA.PANDAMLTEST')
        else:
            logging.error(f"Processing failed: {result}")
            error_df = r.copy()  # Copy the original DataFrame
            error_df['ERROR'] = result  # Add the error message as a new column
            # Remove any unnecessary columns
            error_df = error_df[['JEDITASKID', 'ERROR']]
            # Add dummy columns if necessary to match the schema of the main table
            for col in cols_to_write:
                if col not in error_df.columns:
                    error_df[col] = None
            output_db.write_data(error_df, 'ATLAS_PANDA.PANDAMLTEST')
        print("Next Trial")
        print(result)

        # # Add a 10-minute delay here
        # print("Waiting for 10 minutes before processing the next task...")
        # time.sleep(4)  # Reduced delay for testing
        # # "Wake up" actions
        # print("Waking up after 10 minutes sleep")
        # logging.info("Resuming execution after sleep period")
        # # You can add any other actions you want to perform after waking up here

    print("All tasks processed")
    input_db.close_connection()
