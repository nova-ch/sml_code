# src/scout_ml_package/train_model.py

import pandas as pd
import joblib
import re
import numpy as np
from keras.layers import TFSMLayer
from scout_ml_package.data import (
    HistoricalDataProcessor,
    DataSplitter,
    CategoricalEncoder,
)
from scout_ml_package.model.model_pipeline import (
    TrainingPipeline,
)  # ,ModelLoader
from scout_ml_package.utils import ErrorMetricsPlotter


# Assuming you have a HistoricalDataProcessor instance and you want to get the merged_data
# processor = HistoricalDataProcessor(task_data_path='path/to/task_data.parquet',
#                                      additional_data_path='path/to/additional_data.parquet')


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

    df["CTIME"] = np.where(
        df["CPUTIMEUNIT"] == "mHS06sPerEvent",
        df["CPUTIME"] / 1000,
        np.where(df["CPUTIMEUNIT"] == "HS06sPerEvent", df["CPUTIME"], None),
    )
    df['CTIME'] = df['CTIME'].astype('float64')

    KEEP_F_TAG = [
        "Athena",
        "AnalysisBase",
        "AtlasOffline",
        "AthAnalysis",
        "AthSimulation",
        "MCProd",
        "AthGeneration",
        "AthDerivation",
    ]
    KEEP_P_TAG = [
        "jedi-run",
        "deriv",
        "athena-trf",
        "jedi-athena",
        "simul",
        "pile",
        "merge",
        "evgen",
        "reprocessing",
        "recon",
        "eventIndex",
    ]

    df["P"] = df["P"].apply(lambda x: x if x in KEEP_P_TAG else "others")
    df["F"] = df["F"].apply(lambda x: x if x in KEEP_F_TAG else "others")
    return df


base_path = "/data/model-data/"
categorical_features = ["PRODSOURCELABEL", "P", "F", "CORE"]
data = pd.read_parquet("/data/model-data/merged_files/c_task.parquet")
dataset = pd.read_parquet("/data/model-data/merged_files/c_data.parquet")
ceff = pd.read_parquet("/data/model-data/merged_files/c_eff.parquet")
df_ = pd.merge(data, dataset, on="JEDITASKID", how="right")
df_ = pd.merge(df_, ceff, on="JEDITASKID", how="left")

# # task_train_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/training_historial.parquet'
# task_train_data_path = (
#     "/data/model-data/merged_files/training_historial.parquet"
# )
# processor = HistoricalDataProcessor(task_train_data_path)

# # task_new_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/new_historial.parquet'
# task_new_data_path = "/data/model-data/merged_files/new_historial.parquet"
# new_preprocessor = HistoricalDataProcessor(task_new_data_path)
# Filter the data
# training_data = processor.filtered_data()
# future_data = new_preprocessor.filtered_data()

df_ = preprocess_data(df_)
df_ = df_[df_["CPUTIMEUNIT"] == "mHS06sPerEvent"].copy()
training_data = df_.sample(frac=0.7, random_state=42)
future_data = df_[
    ~df_.index.isin(training_data.index)
]  # Get the remaining rows


#################################################################################################################
#################################################################################################################
# Prepare train test dataset for all the models in sequence RAMCOUNT-> cputime_HS -> CPU_EFF -> IOINTENSITY
#################################################################################################################
print(df_.shape)
print(training_data.shape)
print(future_data.shape)


target_var = ["CTIME"]

encoder = CategoricalEncoder()
category_list = encoder.get_unique_values(
    training_data, categorical_features
)  # Get unique values
print(category_list)
# Define the columns you want to select
selected_columns = [
    "JEDITASKID",
    "PRODSOURCELABEL",
    "P",
    "F",
    "CPUTIMEUNIT",
    "CORE",
    "TOTAL_NFILES",
    "TOTAL_NEVENTS",
    "DISTINCT_DATASETNAME_COUNT",
    "RAMCOUNT",
    "CTIME",
    "CPU_EFF",
    "IOINTENSITY",
]


# Further filter the training data based on specific criteria
training_data = training_data[
    (training_data["PRODSOURCELABEL"].isin(["user", "managed"]))
    & (training_data["CTIME"] > 20)
    & (training_data["CTIME"] < 4000)
    & (training_data["RAMCOUNT"] < 8000)
    & (training_data["RAMCOUNT"] > 10)
    #& (training_data["CPUTIMEUNIT"] == "HS06sPerEvent")
]
categorical_features = ["PRODSOURCELABEL", "P", "F", "CORE"]
print(training_data.shape)
splitter = DataSplitter(training_data, selected_columns)
train_df, test_df = splitter.split_data(test_size=0.15)

# Preprocess the data

numerical_features = [
    "TOTAL_NFILES",
    "TOTAL_NEVENTS",
    "DISTINCT_DATASETNAME_COUNT",
    "RAMCOUNT"
]
features = numerical_features + categorical_features

# @@@@@@@@@@@@@@@@@@@@@@@@@@@------------------
print("Pipeline Test")

pipeline = TrainingPipeline(
    numerical_features, categorical_features, category_list, target_var
)

(
    processed_train_data,
    processed_test_data,
    processed_future_data,
    encoded_columns,
    fitted_scalar,
) = pipeline.preprocess_data(train_df, test_df, future_data)

features_to_train = encoded_columns + numerical_features

tuned_model = pipeline.train_model(
    processed_train_data,
    processed_test_data,
    features_to_train,
    "build_cputime_low",
    epoch=1,
    batch=200,
)  # build_cputime
predictions, y_pred = pipeline.regression_prediction(
    tuned_model, processed_future_data, features_to_train
)


model_seq = "3"
target_name = "cputime_high"

model_storage_path = (
    # Define the storage path
    f"/data/model-data/ModelStorage/model{model_seq}/"
)
model_name = f"model{model_seq}_{target_name}"  # Define the model name
# 'my_plots'  # Optional: specify a custom plots directory
plot_directory_name = f"/data/model-data/ModelStorage/plots/model{model_seq}"

joblib.dump(fitted_scalar, f"{model_storage_path}/scaler.pkl")

model_full_path = model_storage_path + model_name
tuned_model.export(model_full_path)

# Specifying custom column names when instantiating the class
actual_column_name = "CTIME"  # Change this to match your actual column name
predicted_column_name = (
    "Predicted_CTIME"  # Change this to match your predicted column name
)
predictions = predictions.dropna()
# Create an instance of the ErrorMetricsPlotter class
plotter = ErrorMetricsPlotter(
    predictions,
    actual_column=actual_column_name,
    predicted_column=predicted_column_name,
    plot_directory=plot_directory_name,
)

# Print error metrics
plotter.print_metrics()
# Plot the metrics
plotter.plot_metrics()


print(model_full_path)
model = TFSMLayer(model_full_path, call_endpoint="serving_default")
predictions = model(processed_future_data[features_to_train])
print(predictions)
