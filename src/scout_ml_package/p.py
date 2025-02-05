# src/scout_ml_package/prediction_pipeline.py
import time
import logging
from scout_ml_package.utils.demo import DummyData, DataValidator, FakeListener
from scout_ml_package.model.model_pipeline import (
    ModelManager,
    PredictionPipeline,
)
from scout_ml_package.data.fetch_db_data import DatabaseFetcher

# Configure logging only once at the start of your script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add a single handler
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# Define acceptable ranges for each prediction
acceptable_ranges = {
    # Adjust these ranges based on your domain knowledge
    "RAMCOUNT": (100, 10000),
    "cputime_HS": (0.4, 10000),
    "CPU_EFF": (0, 100),
}


start_time = time.time()


def get_prediction(model_manager, r):
    start_time = time.time()

    try:
        if r is not None:
            jeditaskid = r["JEDITASKID"].values[0]
            processor = PredictionPipeline(model_manager)
            base_df = processor.preprocess_data(r)

            # Model 1: RAMCOUNT
            features = (
                ["JEDITASKID"]
                + processor.numerical_features
                + processor.category_sequence
            )
            base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model(
                "1", features, base_df
            )
            DataValidator.check_predictions(
                base_df, "RAMCOUNT", acceptable_ranges
            )

            # Model 2 and 3: cputime_HS
            processor.numerical_features.append("RAMCOUNT")
            features = (
                ["JEDITASKID"]
                + processor.numerical_features
                + processor.category_sequence
            )

            if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
                base_df.loc[:, "cputime_HS"] = (
                    processor.make_predictions_for_model(
                        "2", features, base_df
                    )
                )
            else:
                base_df.loc[:, "cputime_HS"] = (
                    processor.make_predictions_for_model(
                        "3", features, base_df
                    )
                )
            DataValidator.check_predictions(
                base_df, "cputime_HS", acceptable_ranges
            )

            # Model 4: CPU_EFF
            processor.numerical_features.append("cputime_HS")
            features = (
                ["JEDITASKID"]
                + processor.numerical_features
                + processor.category_sequence
            )
            base_df.loc[:, "CPU_EFF"] = processor.make_predictions_for_model(
                "4", features, base_df
            )
            DataValidator.check_predictions(
                base_df, "CPU_EFF", acceptable_ranges
            )

            # Model 5: IOINTENSITY
            processor.numerical_features.append("CPU_EFF")
            features = (
                ["JEDITASKID"]
                + processor.numerical_features
                + processor.category_sequence
            )
            base_df.loc[:, "IOINTENSITY"] = (
                processor.make_predictions_for_model("5", features, base_df)
            )

            logging.info(
                f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
            )
            return base_df

        else:
            logging.error("Failed to process: Input data is None")
            return None

    except ValueError as ve:
        logging.error(
            f"Check failed for JEDITASKID {r['JEDITASKID'].values[0] if r is not None else 'Unknown'}: {ve}"
        )
        return None

    except Exception as e:
        logging.error(
            f"Error processing JEDITASKID {r['JEDITASKID'].values[0] if r is not None else 'Unknown'}: {str(e)}"
        )
        return None


if __name__ == "__main__":
    df = DummyData.fetch_data()
    # base_path = "/Users/tasnuvachowdhury/Desktop/PROD/pandaml-test/src/"
    base_path = "/data/model-data/"  # "/data/test/"
    model_manager = ModelManager(base_path)
    model_manager.load_models()

    db_fetcher = DatabaseFetcher()

    sample_tasks = [27766704, 27746332]
    listener = FakeListener(sample_tasks, delay=3)  # Pass delay here
    for (
        jeditaskid
    ) in listener.demo_task_listener():  # No arguments needed here
        print(f"Received JEDITASKID: {jeditaskid}")
        r = db_fetcher.fetch_task_param(jeditaskid)
        # r = df[df['JEDITASKID'] == jeditaskid].copy()
        print(r)
        result = get_prediction(model_manager, r)
        if result is not None:
            logging.info("Processing completed successfully")
        else:
            logging.error("Processing failed due to invalid results or errors")
        print("Next Trial")
        print(result)

        # Add a 10-minute delay here
        print("Waiting for 10 minutes before processing the next task...")
        time.sleep(4)  # Reduced delay for testing
        # "Wake up" actions
        print("Waking up after 10 minutes sleep")
        logging.info("Resuming execution after sleep period")
        # You can add any other actions you want to perform after waking up here

    print("All tasks processed")
    db_fetcher.close_connection()
