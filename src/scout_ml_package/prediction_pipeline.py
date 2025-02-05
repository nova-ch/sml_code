# src/scout_ml_package/prediction_pipeline.py
from scout_ml_package.data.fetch_db_data import get_db_connection
from scout_ml_package.model.model_pipeline import ModelHandlerInProd
import pandas as pd
import numpy as np
import re
import logging
import time

#########
import warnings

# Ignore all warnings (not recommended for production code)
warnings.filterwarnings("ignore")

# Or, to ignore specific warnings (like FutureWarnings)
warnings.filterwarnings("ignore", category=FutureWarning)


# Example function to preprocess data
def preprocess_data(df):
    # Convert PROCESSINGTYPE to 'P'
    def convert_processingtype(processingtype):
        if processingtype is not None and re.search(r"-.*-", processingtype):
            return "-".join(processingtype.split("-")[-2:])
        return processingtype

    # Convert TRANSHOME to 'F'
    def convert_transhome(transhome):
        # Check if transhome is None
        if transhome is None:
            return None  # or handle as needed

        if "AnalysisBase" in transhome:
            return "AnalysisBase"
        elif "AnalysisTransforms-" in transhome:
            # Extract the part after 'AnalysisTransforms-'
            part_after_dash = transhome.split("-")[1]
            return part_after_dash.split("_")[
                0
            ]  # Assuming you want the first part before any underscore
        elif "/" in transhome:
            # Handle cases like AthGeneration/2022-11-09T1600
            return transhome.split("/")[0]
        else:
            # For all other cases, split by '-', return the first segment
            return transhome.split("-")[0]

    # Convert CORECOUNT to 'Core'
    def convert_corecount(corecount):
        return "S" if corecount == 1 else "M"

    # Apply transformations
    df["P"] = df["PROCESSINGTYPE"].apply(convert_processingtype)
    df["F"] = df["TRANSHOME"].apply(convert_transhome)
    df["CORE"] = df["CORECOUNT"].apply(convert_corecount)

    # Return selected columns
    numerical_features = [
        "TOTAL_NFILES",
        "TOTAL_NEVENTS",
        "DISTINCT_DATASETNAME_COUNT",
    ]
    categorical_features = ["PRODSOURCELABEL", "P", "F", "CPUTIMEUNIT", "CORE"]
    ["JEDITASKID"] + numerical_features + categorical_features

    return df  # [features]


# Example data fetching process (to be replaced with your actual data source)
def fetch_data():
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


start_time = time.time()
# Fetch and preprocess the data
df = fetch_data()


def process_row(row, df):
    try:
        if row["CPUTIMEUNIT"] == "mHS06sPerEvent":
            mhc = ModelHandlerInProd(
                model_sequence="2", target_name="cputime_low"
            )
            print("Model 2 Loaded")
            temp_df_single_row = df[df["CPUTIMEUNIT"] == "mHS06sPerEvent"]

        else:
            mhc = ModelHandlerInProd(
                model_sequence="3", target_name="cputime_high"
            )
            temp_df_single_row = df[df["CPUTIMEUNIT"] == "HS06sPerEvent"]

            print("Model 3 Loaded")

        # Load model and scaler
        mhc.load_model_and_scaler()

        return (
            mhc,
            temp_df_single_row,
        )  # Return both the handler and the filtered DataFrame
    except Exception as e:
        print(f"Error in process_row for row {row}: {e}")
        return None, None  # Return None for both if there's an error


###############

# Set up logging for the application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def make_predictions_for_model(
    mh,
    features,
    numerical_features,
    category_sequence,
    unique_elements_categories,
    input_df,
):
    """Makes predictions using a given model handler and input DataFrame."""
    try:
        # Preprocess data
        processed_data, features_to_train = mh.preprocess_data(
            input_df[features],
            numerical_features,
            category_sequence,
            unique_elements_categories,
        )
        print(processed_data.head())
        return mh.make_predictions(processed_data, features_to_train)
    except Exception as e:
        logging.error(
            f"Error processing data with model sequence {mh.model_sequence}: {e}"
        )
        return None


# Timing the execution
start_time = time.time()
# Preprocessing data
base_df = preprocess_data(df)

# Model 1
target_var1 = "RAMCOUNT"
mh1 = ModelHandlerInProd(model_sequence="1", target_name="ramcount")
mh1.load_model_and_scaler()

numerical_features = [
    "TOTAL_NFILES",
    "TOTAL_NEVENTS",
    "DISTINCT_DATASETNAME_COUNT",
]
category_sequence = ["PRODSOURCELABEL", "P", "F", "CPUTIMEUNIT", "CORE"]
unique_elements_categories = [
    ["managed", "user"],
    [
        "deriv",
        "jedi-run",
        "jedi-athena",
        "eventIndex",
        "simul",
        "evgen",
        "merge",
        "pile",
        "athena-trf",
        "recon",
        "reprocessing",
        "overlay",
        "run-evp",
        "trf-grl",
        "gangarobot-athena",
        "athena-evp",
        "digit",
    ],
    [
        "AthDerivation",
        "AnalysisBase",
        "Athena",
        "AthGeneration",
        "AtlasProduction",
        "AtlasOffline",
        "AthAnalysis",
        "AnalysisTransforms",
        "MCProd",
        "AthSimulation",
        "AtlasProd1",
        "AtlasDerivation",
        "AnalysisTop",
    ],
    ["HS06sPerEvent", "mHS06sPerEvent"],
    ["M", "S"],
]
features = ["JEDITASKID"] + numerical_features + category_sequence
base_df[target_var1] = make_predictions_for_model(
    mh1,
    features,
    numerical_features,
    category_sequence,
    unique_elements_categories,
    base_df,
)
print(base_df)


# Model 2
target_var2 = "cputime_HS"
numerical_features.append(target_var1)  # Include RAMCOUNT from Model 1
features = ["JEDITASKID"] + numerical_features + category_sequence

predicted_values_list = []
for index, row in base_df.iterrows():
    mhc, temp_df_single_row = process_row(row, base_df)
    if mhc is None:
        logging.warning(f"Skipping row {index} due to model loading error.")
        continue

    predicted_value = make_predictions_for_model(
        mhc,
        features,
        numerical_features,
        category_sequence,
        unique_elements_categories,
        temp_df_single_row,
    )
    if predicted_value is not None:
        predicted_values_list.append(predicted_value)

base_df[target_var2] = np.array(predicted_values_list)

# Model 4
target_var3 = "CPU_EFF"
mh4 = ModelHandlerInProd(model_sequence="4", target_name="cpu_eff")
mh4.load_model_and_scaler()
numerical_features.append(target_var2)
features = ["JEDITASKID"] + numerical_features + category_sequence
base_df[target_var3] = make_predictions_for_model(
    mh4,
    features,
    numerical_features,
    category_sequence,
    unique_elements_categories,
    base_df,
)

# Model 5
target_var4 = "IOINTENSITY"
mh5 = ModelHandlerInProd(model_sequence="5", target_name="io")
mh5.load_model_and_scaler()
numerical_features.append(target_var3)
features = ["JEDITASKID"] + numerical_features + category_sequence
base_df[target_var4] = make_predictions_for_model(
    mh5,
    features,
    numerical_features,
    category_sequence,
    unique_elements_categories,
    base_df,
)
print(base_df)


# Execution time logging
end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Execution Time: {execution_time:.2f} seconds")


def test_db_connection(conn):
    try:
        # Simple test query
        cursor = conn.cursor()
        # For Oracle, 'dual' is a dummy table
        cursor.execute("SELECT 1 FROM dual")
        result = cursor.fetchone()
        print("Database connection test successful:", result)
    except Exception as e:
        print(f"Database connection test failed: {e}")
    finally:
        cursor.close()


if __name__ == "__main__":
    try:
        # Get database connection
        conn = get_db_connection()
        conn = get_db_connection()
        test_db_connection(conn)
        # # List of task IDs to fetch data for
        # jeditaskids = [27766704, 27766716, 27766187, 27746332]
        #
        # # Fetch data and display results
        # df = fetch_task_param(jeditaskids, conn)
        # print(df)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if "conn" in locals() and conn:
            conn.close()
