import oracledb
import pandas as pd
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
import sqlite3

base_path = "/data/model-data/" 
input_db = DatabaseFetcher('database')
output_db = DatabaseFetcher('output_database')

sample_tasks = [27766704, 27746332]
r = input_db.fetch_task_param(sample_tasks)

print(r)

query = """
SELECT * FROM atlas_panda.pandamltest
"""
df = pd.read_sql(query, con=output_db.get_connection())
print(df)

# # Create a sample DataFrame to write
# sample_data = pd.DataFrame({
#     'column1': [1, 2, 3],
#     'column2': ['a', 'b', 'c']
# })

# # Write the sample data to the output database
# output_db.write_data(sample_data, 'output_table_name')
