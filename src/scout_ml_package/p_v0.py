import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# src/scout_ml_package/prediction_pipeline.py
from scout_ml_package.model.model_pipeline import  ModelHandlerInProd
import pandas as pd
from scout_ml_package.data.fetch_db_data import fetch_task_param
import numpy as np
import re, logging, time, warnings, random
# from scout_ml_package.utils.logger import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging

# Configure logging only once at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a single handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

# # Example of using the updated method with a specific base path
# handler = ModelHandlerInProd(model_sequence="1", target_name="ramcount")
# handler.load_model_and_scaler(base_path="/absolute/path/to/models")
#
# # Example of using the method without specifying base_path (defaults to current working directory)
# handler.load_model_and_scaler()


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
            model.load_model_and_scaler(base_path="/Users/tasnuvachowdhury/Desktop/PROD/pandaml-test/src/")
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

        return df

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




def get_prediction(model_manager, r):
    start_time = time.time()

    try:
        if r is not None:
            jeditaskid = r['JEDITASKID'].values[0]
            processor = PredictionPipeline(model_manager)
            base_df = processor.preprocess_data(r)

            # Model 1: RAMCOUNT
            features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
            base_df.loc[:, 'RAMCOUNT'] = processor.make_predictions_for_model("1", features, base_df)
            check_predictions(base_df, 'RAMCOUNT', acceptable_ranges['RAMCOUNT'])

            # Model 2 and 3: cputime_HS
            processor.numerical_features.append('RAMCOUNT')
            features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence

            if base_df['CPUTIMEUNIT'].values[0] == 'mHS06sPerEvent':
                base_df.loc[:, 'cputime_HS'] = processor.make_predictions_for_model("2", features, base_df)
            else:
                base_df.loc[:, 'cputime_HS'] = processor.make_predictions_for_model("3", features, base_df)

            check_predictions(base_df, 'cputime_HS', acceptable_ranges['cputime_HS'])

            # Model 4: CPU_EFF
            processor.numerical_features.append('cputime_HS')
            features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
            base_df.loc[:, 'CPU_EFF'] = processor.make_predictions_for_model("4", features, base_df)
            check_predictions(base_df, 'CPU_EFF', acceptable_ranges['CPU_EFF'])

            # Model 5: IOINTENSITY
            processor.numerical_features.append('CPU_EFF')
            features = ['JEDITASKID'] + processor.numerical_features + processor.category_sequence
            base_df.loc[:, 'IOINTENSITY'] = processor.make_predictions_for_model("5", features, base_df)

            logging.info(f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds")
            return base_df

        else:
            logging.error("Failed to process: Input data is None")
            return None

    except ValueError as ve:
        logging.error(f"Check failed for JEDITASKID {r['JEDITASKID'].values[0] if r is not None else 'Unknown'}: {ve}")
        return None

    except Exception as e:
        logging.error(f"Error processing JEDITASKID {r['JEDITASKID'].values[0] if r is not None else 'Unknown'}: {str(e)}")
        return None


if __name__ == "__main__":
    df = fetch_data()
    model_manager = ModelManager()
    model_manager.load_models()

    sample_tasks = [27766704, 27746332]
    for jeditaskid in demo_task_listener(sample_tasks, delay=3):
        print(f"Received JEDITASKID: {jeditaskid}")
        r = df[df['JEDITASKID'] == jeditaskid].copy()
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