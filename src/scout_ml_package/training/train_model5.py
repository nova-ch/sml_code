# src/scout_ml_package/train_model.py

#from scout_ml_package.model import MultiOutputModel, TrainedModel, PredictionVisualizer
#from scout_ml_package.model.base_model import ModelTrainer, ModelPipeline
from scout_ml_package.data import HistoricalDataProcessor, DataSplitter, ModelTrainingInput, CategoricalEncoder, TrainingDataPreprocessor, NewDataPreprocessor
from scout_ml_package.model.model_pipeline import TrainingPipeline#,ModelLoader
from scout_ml_package.utils import ClassificationMetricsPlotter
import joblib

# Assuming you have a HistoricalDataProcessor instance and you want to get the merged_data
# processor = HistoricalDataProcessor(task_data_path='path/to/task_data.parquet',
#                                      additional_data_path='path/to/additional_data.parquet')

task_train_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/training_historial.parquet'
processor = HistoricalDataProcessor(task_train_data_path)

task_new_data_path = '/Users/tasnuvachowdhury/Desktop/projects/draft_projects/SML/local_data/new_historial.parquet'
new_preprocessor = HistoricalDataProcessor(task_new_data_path)
# Filter the data
training_data = processor.filtered_data()
future_data = new_preprocessor.filtered_data()

#################################################################################################################
#################################################################################################################
# Prepare train test dataset for all the models in sequence RAMCOUNT-> cputime_HS -> CPU_EFF -> IOINTENSITY
#################################################################################################################

target_var = ['IOINTENSITY']


categorical_features = ['PRODSOURCELABEL', 'P', 'F', 'CPUTIMEUNIT', 'CORE']
encoder = CategoricalEncoder()
category_list = encoder.get_unique_values(training_data, categorical_features) # Get unique values
print(category_list)
# Define the columns you want to select
selected_columns = [
    'JEDITASKID', 'PRODSOURCELABEL', 'P', 'F', 'CPUTIMEUNIT', 'CORE',
    'TOTAL_NFILES', 'TOTAL_NEVENTS', 'DISTINCT_DATASETNAME_COUNT',
    'RAMCOUNT', 'cputime_HS', 'CPU_EFF', 'P50','F50', 'IOINTENSITY'
]

# Further filter the training data based on specific criteria
training_data = training_data[
    (training_data['PRODSOURCELABEL'].isin(['user', 'managed'])) &
    (training_data['RAMCOUNT'] > 100) &
    (training_data['RAMCOUNT'] < 6000) &
    (training_data['CPU_EFF'] > 30) &
    (training_data['CPU_EFF'] < 100) &
    ((training_data['cputime_HS'] > 0.5) & (training_data['cputime_HS'] < 4) )|
    ((training_data['cputime_HS'] > 10) & (training_data['cputime_HS'] < 6000))
]

# label_mapping = {'low': 0, 'high': 1}
#
# # Apply the mapping to your target variable
# training_data[target_var] = training_data[target_var].map(label_mapping)
# future_data[target_var] = future_data[target_var].map(label_mapping)
#
import pandas as pd
# Handle non-mapping cases and set them as NaN
training_data[target_var] = training_data[target_var].replace({'low': 0, 'high': 1})
future_data[target_var] = future_data[target_var].replace({'low': 0, 'high': 1})

# If you want to be explicit about NaNs
training_data[target_var] = training_data[target_var].astype('int32')  # For labels as integers
future_data[target_var] = future_data[target_var].astype('int32')

print(training_data.shape)
splitter = DataSplitter(training_data, selected_columns)
train_df, test_df = splitter.split_data(test_size=0.15)

# Preprocess the data
numerical_features = ['TOTAL_NFILES', 'TOTAL_NEVENTS', 'DISTINCT_DATASETNAME_COUNT', 'RAMCOUNT', 'cputime_HS', 'CPU_EFF']
categorical_features = ['PRODSOURCELABEL', 'P', 'F', 'CPUTIMEUNIT', 'CORE']
features = numerical_features + categorical_features

##########@@@@@@@@@@@@@@@@@@@@@@@@@@@------------------
print("Pipeline Test")

print(training_data['IOINTENSITY'])


pipeline = TrainingPipeline(numerical_features, categorical_features, category_list, target_var)

processed_train_data, processed_test_data, processed_future_data, encoded_columns, fitted_scalar  = pipeline.preprocess_data(train_df,
                                                                                                             test_df,
                                                                                                             future_data)

features_to_train = encoded_columns + numerical_features

tuned_model = pipeline.train_classification_model(processed_train_data, processed_test_data, features_to_train,
                                       'build_io', epoch=1, batch=128) #build_cputime
predictions, y_pred = pipeline.classification_prediction(tuned_model, processed_future_data, features_to_train)


print(predictions.head())
# model_storage_path = "ModelStorage/model5/"  # Define the storage path
# model_name = "model5_io"  # Define the model name


model_seq = "5"
target_name = "io"
model_storage_path = f"ModelStorage/model{model_seq}/"  # Define the storage path
model_name = f"model{model_seq}_{target_name}"  # Define the model name
plot_directory_name = f'ModelStorage/plots/model{model_seq}'#'my_plots'  # Optional: specify a custom plots directory
model_full_path = model_storage_path+model_name




joblib.dump(fitted_scalar, f'{model_storage_path}/scaler.pkl')


# Save the model using ModelHandler
#pipeline.ModelHandler.save_model(tuned_model, model_storage_path, model_name, format='keras')

tuned_model.export(model_full_path)

# Specifying custom column names when instantiating the class
actual_column_name = 'IOINTENSITY'  # Change this to match your actual column name
predicted_column_name = 'Predicted_IOINTENSITY'  # Change this to match your predicted column name
# plot_directory_name = 'ModelStorage/plots/model5'#'my_plots'  # Optional: specify a custom plots directory

# Create an instance of the ErrorMetricsPlotter class
plotter = ClassificationMetricsPlotter(predictions, actual_column=actual_column_name, predicted_column=predicted_column_name, plot_directory=plot_directory_name)

plotter.calculate_and_print_metrics()
plotter.plot_confusion_matrix()

#print_error_metrics(predictions, actual_column='cputime_HS', predicted_column='Predicted_cputime_HS', plot_directory=model_storage_path)

from keras.layers import TFSMLayer
import tensorflow as tf
model = TFSMLayer(model_full_path, call_endpoint='serving_default')
predictions = model(processed_future_data[features_to_train])
print(predictions)
