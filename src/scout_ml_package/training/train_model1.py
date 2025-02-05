# src/scout_ml_package/train_model.py

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
import joblib

# Assuming you have a HistoricalDataProcessor instance and you want to get the merged_data
# processor = HistoricalDataProcessor(task_data_path='path/to/task_data.parquet',
#                                      additional_data_path='path/to/additional_data.parquet')
base_path = "/data/model-data/"
# task_train_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/training_historial.parquet'
task_train_data_path = (
    "/data/model-data/merged_files/training_historial.parquet"
)
processor = HistoricalDataProcessor(task_train_data_path)

# task_new_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/new_historial.parquet'
task_new_data_path = "/data/model-data/merged_files/new_historial.parquet"
new_preprocessor = HistoricalDataProcessor(task_new_data_path)
# Filter the data
training_data = processor.filtered_data()
future_data = new_preprocessor.filtered_data()

#################################################################################################################
#################################################################################################################
# Prepare train test dataset for all the models in sequence RAMCOUNT-> cputime_HS -> CPU_EFF -> IOINTENSITY
#################################################################################################################

target_var = ["RAMCOUNT"]


categorical_features = ["PRODSOURCELABEL", "P", "F", "CPUTIMEUNIT", "CORE"]
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
    "cputime_HS",
    "CPU_EFF",
    "P50",
    "F50",
    "IOINTENSITY",
]

# Further filter the training data based on specific criteria
training_data = training_data[
    (training_data["PRODSOURCELABEL"].isin(["user", "managed"]))
    & (training_data["RAMCOUNT"] > 100)
    & (training_data["RAMCOUNT"] < 6000)
    & (training_data["CPU_EFF"] > 30)
    & (training_data["CPU_EFF"] < 100)
    & ((training_data["cputime_HS"] > 0.5) & (training_data["cputime_HS"] < 4))
    | (
        (training_data["cputime_HS"] > 10)
        & (training_data["cputime_HS"] < 3000)
    )
]

print(training_data.shape)
splitter = DataSplitter(training_data, selected_columns)
train_df, test_df = splitter.split_data(test_size=0.15)

# Preprocess the data

numerical_features = [
    "TOTAL_NFILES",
    "TOTAL_NEVENTS",
    "DISTINCT_DATASETNAME_COUNT",
]
categorical_features = ["PRODSOURCELABEL", "P", "F", "CPUTIMEUNIT", "CORE"]
features = numerical_features + categorical_features

# @@@@@@@@@@@@@@@@@@@@@@@@@@@------------------
print("Pipeline Test")

# numerical_features = [...]  # Define your numerical features
# categorical_features = [...]  # Define your categorical features
# category_list = [...]  # Define your categories
# model_target = [...]  # Define model target columns

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
    "test_build",
    epoch=100,
    batch=128,
)  # build_cputime
predictions, y_pred = pipeline.regression_prediction(
    tuned_model, processed_future_data, features_to_train
)

model_seq = "1"
target_name = "ramcount"
model_storage_path = (
    # Define the storage path
    f"/data/model-data/ModelStorage/model{model_seq}/"
)
model_name = f"model{model_seq}_{target_name}"  # Define the model name
# 'my_plots'  # Optional: specify a custom plots directory
plot_directory_name = f"/data/model-data/ModelStorage/plots/model{model_seq}"

joblib.dump(fitted_scalar, f"{model_storage_path}/scaler.pkl")
model_full_path = model_storage_path + model_name
# Save the model using ModelHandler
# pipeline.ModelHandler.save_model(tuned_model, model_storage_path, model_name, format='keras')

# tuned_model.export('ModelStorage/model2/model2_cputime_low')
tuned_model.export(model_full_path)

# Specifying custom column names when instantiating the class
actual_column_name = "RAMCOUNT"  # Change this to match your actual column name
predicted_column_name = (
    "Predicted_RAMCOUNT"  # Change this to match your predicted column name
)

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

# print_error_metrics(predictions, actual_column='cputime_HS', predicted_column='Predicted_cputime_HS', plot_directory=model_storage_path)


print(model_full_path)
model = TFSMLayer(model_full_path, call_endpoint="serving_default")
predictions = model(processed_future_data[features_to_train])
print(predictions)
