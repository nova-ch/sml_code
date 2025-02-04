# src/scout_ml_package/prediction_pipeline.py
from scout_ml_package.model.model_pipeline import  ModelHandlerInProd
import pandas as pd
from scout_ml_package.data.fetch_db_data import fetch_task_param
import re, logging, time, warnings, random
from scout_ml_package.utils.logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging

# Configure logging only once at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable propagation to prevent duplicate logs
logger.propagate = False

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a single handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
import time

def demo_task_listener(task_ids, delay=3):
    """
    A demo function that simulates a listener sending task IDs with a fixed delay.

    Args:
    task_ids (list): A list of task IDs to send.
    delay (int): The fixed delay in seconds between each task ID.

    Yields:
    int: A task ID from the list.
    """
    for task_id in task_ids:
        # Wait for the specified delay
        time.sleep(delay)
        yield task_id

def fetch_data():
    # Simulate data retrieval
    data = {
        'JEDITASKID': [27766704, 27746332],
        'PRODSOURCELABEL': ['managed', 'user'],
        'PROCESSINGTYPE': ['deriv', 'panda-client-1.4.98-jedi-run'],
        'TRANSHOME': ['AthDerivation-21.2.77.0', 'AnalysisTransforms-AnalysisBase_21.2.197'],
        'CPUTIMEUNIT': ['HS06sPerEvent', 'mHS06sPerEvent'],
        'CORECOUNT': [8, 1],
        'TOTAL_NFILES': [290000, 11237955],
        'TOTAL_NEVENTS': [23, 260],
        'DISTINCT_DATASETNAME_COUNT': [1, 3]
    }
    return pd.DataFrame(data)
start_time = time.time()
# Fetch and preprocess the data
df = fetch_data()

# # Set up logging for the application
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.models = {}

    def load_models(self):
        model_configs = [
            ("1", "ramcount"),
            ("2", "cputime_low"),
            ("3", "cputime_high"),
            ("4", "cpu_eff"),
            ("5", "io")
        ]
        for sequence, target_name in model_configs:
            model = ModelHandlerInProd(model_sequence=sequence, target_name=target_name)
            model.load_model_and_scaler()
            self.models[sequence] = model
        #logger.info("All models loaded successfully")

    def get_model(self, sequence):
        return self.models.get(sequence)


class PredictionPipeline:

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.numerical_features = ['TOTAL_NFILES', 'TOTAL_NEVENTS', 'DISTINCT_DATASETNAME_COUNT']
        self.category_sequence = ['PRODSOURCELABEL', 'P', 'F', 'CPUTIMEUNIT', 'CORE']
        self.unique_elements_categories = [
            ['managed', 'user'],
            ['deriv', 'jedi-run', 'jedi-athena', 'eventIndex', 'simul', 'evgen',
             'merge', 'pile', 'athena-trf', 'recon', 'reprocessing', 'overlay',
             'run-evp', 'trf-grl', 'gangarobot-athena', 'athena-evp', 'digit'],
            ['AthDerivation', 'AnalysisBase', 'Athena', 'AthGeneration', 'AtlasProduction',
             'AtlasOffline', 'AthAnalysis', 'AnalysisTransforms', 'MCProd',
             'AthSimulation', 'AtlasProd1', 'AtlasDerivation', 'AnalysisTop'],
            ['HS06sPerEvent', 'mHS06sPerEvent'],
            ['M', 'S']
        ]

    def preprocess_data(self, df):
        # Convert PROCESSINGTYPE to 'P'
        def convert_processingtype(processingtype):
            if processingtype is not None and re.search(r'-.*-', processingtype):
                return '-'.join(processingtype.split('-')[-2:])
            return processingtype

        # Convert TRANSHOME to 'F'
        def convert_transhome(transhome):
            # Check if transhome is None
            if transhome is None:
                return None  # or handle as needed

            if 'AnalysisBase' in transhome:
                return 'AnalysisBase'
            elif 'AnalysisTransforms-' in transhome:
                # Extract the part after 'AnalysisTransforms-'
                part_after_dash = transhome.split('-')[1]
                return part_after_dash.split('_')[0]  # Assuming you want the first part before any underscore
            elif '/' in transhome:
                # Handle cases like AthGeneration/2022-11-09T1600
                return transhome.split('/')[0]
            else:
                # For all other cases, split by '-', return the first segment
                return transhome.split('-')[0]

        # Convert CORECOUNT to 'Core'
        def convert_corecount(corecount):
            return 'S' if corecount == 1 else 'M'

        # Apply transformations
        df['P'] = df['PROCESSINGTYPE'].apply(convert_processingtype)
        df['F'] = df['TRANSHOME'].apply(convert_transhome)
        df['CORE'] = df['CORECOUNT'].apply(convert_corecount)

        # Return selected columns
        numerical_features = ['TOTAL_NFILES', 'TOTAL_NEVENTS', 'DISTINCT_DATASETNAME_COUNT']
        categorical_features = ['PRODSOURCELABEL', 'P', 'F', 'CPUTIMEUNIT', 'CORE']
        features = ['JEDITASKID'] + numerical_features + categorical_features

        return df  # [features]

    def make_predictions_for_model(self, model_sequence, features, input_df):
        try:
            mh = self.model_manager.get_model(model_sequence)
            if mh is None:
                raise ValueError(f"Model with sequence {model_sequence} not found")

            processed_data, features_to_train = mh.preprocess_data(
                input_df[features],
                self.numerical_features,
                self.category_sequence,
                self.unique_elements_categories
            )
            return mh.make_predictions(processed_data, features_to_train)
        except Exception as e:
            logger.error(f"Error processing data with model sequence {model_sequence}: {e}")
            return None

    def process_row(self, row, df):
        try:
            if row['CPUTIMEUNIT'] == 'mHS06sPerEvent':
                model_sequence = "2"
                temp_df = df[df['CPUTIMEUNIT'] == 'mHS06sPerEvent']
            else:
                model_sequence = "3"
                temp_df = df[df['CPUTIMEUNIT'] == 'HS06sPerEvent']

            return model_sequence, temp_df
        except Exception as e:
            logger.error(f"Error in process_row for row {row}: {e}")
            return None, None





def main():
    start_time = time.time()
    logger.info("Starting main processing function")

    try:
        # Initialize and load all models
        #logger.info("Initializing and loading models")
        model_manager = ModelManager()
        model_manager.load_models()

        # Fetch and preprocess the data
        logger.info("Fetching and preprocessing data")
        sample_tasks = [27766704, 27766704, 27766187, 27746332, 27766205, 27746332]
        df = fetch_data()
        # Use the demo listener
        for jeditaskid in demo_task_listener(sample_tasks, delay=10):
            print(f"Received JEDITASKID: {jeditaskid}")
            r = df[df['JEDITASKID'] == jeditaskid]
            if r is not None:
                logger.info(f"Successfully processed JEDITASKID: {jeditaskid}")
                processor = PredictionPipeline(model_manager)
                base_df = processor.preprocess_data(df)

                logger.debug(f"Processing completed successfully. Results:\n{base_df['JEDITASKID'].values}")

                # Define acceptable ranges for each prediction
                acceptable_ranges = {
                    'RAMCOUNT': (100, 10000),  # Adjust these ranges based on your domain knowledge
                    'cputime_HS': (0.4, 10000),
                    'CPU_EFF': (0, 100)
                }

                def check_predictions(df, column, range_tuple):
                    min_val, max_val = range_tuple
                    if df[column].min() < min_val or df[column].max() > max_val:
                        raise ValueError(f"Predictions for {column} are outside the acceptable range of {range_tuple}")

                # def process_row_wrapper(row):
                #     model_sequence, temp_df = processor.process_row(row, base_df)
                #     if model_sequence is None:
                #         return None
                #     return processor.make_predictions_for_model(model_sequence, features, temp_df)

                # Model 1: RAMCOUNT
                features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
                base_df['RAMCOUNT'] = processor.make_predictions_for_model("1", features, base_df)
                check_predictions(base_df, 'RAMCOUNT', acceptable_ranges['RAMCOUNT'])

                # Model 2 and 3: cputime_HS
                processor.numerical_features.append('RAMCOUNT')
                features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence

                # with ThreadPoolExecutor() as executor:
                #     futures = [executor.submit(process_row_wrapper, row) for _, row in base_df.iterrows()]
                #     predicted_values = [future.result() for future in as_completed(futures) if
                #                         future.result() is not None]
                #
                processor.numerical_features.append('RAMCOUNT')
                features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence

                predicted_values = []
                for _, row in base_df.iterrows():
                    model_sequence, temp_df = processor.process_row(row, base_df)
                    if model_sequence is not None:
                        prediction = processor.make_predictions_for_model(model_sequence, features, temp_df)
                        predicted_values.append(prediction)

                base_df['cputime_HS'] = np.array(predicted_values)
                check_predictions(base_df, 'cputime_HS', acceptable_ranges['cputime_HS'])



                base_df['cputime_HS'] = np.array(predicted_values)
                check_predictions(base_df, 'cputime_HS', acceptable_ranges['cputime_HS'])

                # Model 4: CPU_EFF
                processor.numerical_features.append('cputime_HS')
                features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
                base_df['CPU_EFF'] = processor.make_predictions_for_model("4", features, base_df)
                check_predictions(base_df, 'CPU_EFF', acceptable_ranges['CPU_EFF'])
                logger.info("Model 3 completed")
                logger.info(base_df.head())

                # Model 5: IOINTENSITY
                processor.numerical_features.append('CPU_EFF')
                features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
                base_df['IOINTENSITY'] = processor.make_predictions_for_model("5", features, base_df)
                # check_predictions(base_df, 'IOINTENSITY', acceptable_ranges['IOINTENSITY'])

                logger.info(f"Processing completed successfully. Results:\n{base_df['JEDITASKID'].values}")
                return base_df  # Return results if all checks pass
                # Here you can do something with the result, like saving it to a database
            else:
                logger.error(f"Failed to process JEDITASKID: {jeditaskid}")






    except ValueError as ve:
        logger.error(f"Check failed: {ve}")
        return None

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return None

    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution Time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    result = main()
    if result is not None:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed due to invalid results or errors")
    print("Next Trial")


