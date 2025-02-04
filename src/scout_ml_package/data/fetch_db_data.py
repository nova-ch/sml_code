# # data/data_fetcher.py
#
# import oracledb
# import pandas as pd
# import configparser
# import os
#
#
# def get_db_connection():
#     # Load configuration
#     config = configparser.ConfigParser()
#
#     # Get the directory of the current script and construct path to config.ini
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     config_path = os.path.join(current_dir,'..', 'config.ini')  # Adjust path if necessary
#
#     print(f"Config file path: {config_path}")
#     print(f"Config file exists: {os.path.exists(config_path)}")
#
#     # Read configuration file
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found at {config_path}")
#
#     config.read(config_path)
#
#     if 'database' not in config:
#         raise KeyError("'database' section missing in config file")
#
#     # Get database credentials
#     database_user = config['database'].get('user')
#     database_password = config['database'].get('password')
#     dsn = config['database'].get('dsn')
#
#     # if not all([database_user, database_password, dsn]):
#     #     raise KeyError("Missing required database credentials in [database] section")
#     #
#     # # Establish connection
#     # conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn)
#     # Establish connection with a timeout
#     try:
#         # Establish connection with tcp_connect_timeout
#         conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn,
#                                 tcp_connect_timeout=20)  # 20 seconds timeout
#         return conn
#
#     except Exception as e:
#         print(f"Failed to connect to the database: {e}")
#
#
# def fetch_task_param(jeditaskids, conn):
#     if not isinstance(jeditaskids, list):
#         jeditaskids = [jeditaskids]
#
#     # Create a string for SQL IN clause
#     jeditaskid_str = ', '.join(map(str, jeditaskids))
#
#     # Combined SQL query
#     query = f"""
#     SELECT
#         jt.jeditaskid,
#         jt.prodsourcelabel,
#         jt.processingtype,
#         jt.transhome,
#         jt.transpath,
#         jt.cputimeunit,
#         jt.corecount,
#         SUM(jd.NFILES) AS total_nfiles,
#         SUM(jd.NEVENTS) AS total_nevents,
#         COUNT(jd.DATASETNAME) AS datasetname_count
#     FROM
#         atlas_panda.jedi_tasks jt
#     LEFT JOIN
#         atlas_panda.jedi_datasets jd ON jt.jeditaskid = jd.jeditaskid AND jd.TYPE = 'input'
#     WHERE
#         jt.jeditaskid IN ({jeditaskid_str}) AND jd.type = 'input'
#     GROUP BY
#         jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.taskname, jt.corecount
#     """
#
#     # Execute the combined query and return the resulting DataFrame
#     df = pd.read_sql(query, con=conn)
#     return df
#
#


import oracledb
import pandas as pd
import configparser
import os


class DatabaseFetcher:
    def __init__(self):
        self.config = self.load_config()
        self.conn = self.get_db_connection()

    def load_config(self):
        # Load configuration
        config = configparser.ConfigParser()

        # Get the directory of the current script and construct path to config.ini
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config.ini')  # Adjust path if necessary

        print(f"Config file path: {config_path}")
        print(f"Config file exists: {os.path.exists(config_path)}")

        # Read configuration file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config.read(config_path)

        if 'database' not in config:
            raise KeyError("'database' section missing in config file")

        return config

    def get_db_connection(self):
        # Get database credentials
        database_user = self.config['database'].get('user')
        database_password = self.config['database'].get('password')
        dsn = self.config['database'].get('dsn')

        try:
            # Establish connection with tcp_connect_timeout
            conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn,
                                    tcp_connect_timeout=20)  # 20 seconds timeout
            return conn

        except Exception as e:
            print(f"Failed to connect to the database: {e}")

    def fetch_task_param(self, jeditaskids):
        if not isinstance(jeditaskids, list):
            jeditaskids = [jeditaskids]

        # Create a string for SQL IN clause
        jeditaskid_str = ', '.join(map(str, jeditaskids))

        # Combined SQL query
        query = f"""
        SELECT 
            jt.jeditaskid,
            jt.prodsourcelabel,
            jt.processingtype,
            jt.transhome,
            jt.transpath,
            jt.cputimeunit,
            jt.corecount,
            SUM(jd.NFILES) AS total_nfiles,
            SUM(jd.NEVENTS) AS total_nevents,
            COUNT(DISTINCT jd.DATASETNAME) AS distinct_datasetname_count
        FROM
            atlas_panda.jedi_tasks jt
        LEFT JOIN
            atlas_panda.jedi_datasets jd ON jt.jeditaskid = jd.jeditaskid AND jd.TYPE = 'input'
        WHERE 
            jt.jeditaskid IN ({jeditaskid_str}) AND jd.type = 'input'
        GROUP BY 
            jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.corecount
        """

        # Execute the combined query and return the resulting DataFrame
        df = pd.read_sql(query, con=self.conn)
        return df


# Example usage
if __name__ == "__main__":

    database_user = "atlas_pandaredwood_r"
    database_password = "N4FFt*4n0McVj+"
    dsn="adcr-s.cern.ch:10121/adcr_panda.cern.ch"
    conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn)
    def fetch_task_param(jeditaskids, conn):
        if not isinstance(jeditaskids, list):
            jeditaskids = [jeditaskids]
    
        # Create a string for SQL IN clause
        jeditaskid_str = ', '.join(map(str, jeditaskids))
    
        # Combined SQL query
        query = f"""
        SELECT 
        jt.jeditaskid,
        jt.prodsourcelabel,
        jt.processingtype,
        jt.transhome,
        jt.transpath,
        jt.cputimeunit,
        jt.corecount,
        SUM(jd.NFILES) AS total_nfiles,
        SUM(jd.NEVENTS) AS total_nevents,
        COUNT(jd.DATASETNAME) AS datasetname_count
        FROM
        atlas_panda.jedi_tasks jt
        LEFT JOIN
        atlas_panda.jedi_datasets jd ON jt.jeditaskid = jd.jeditaskid AND jd.TYPE = 'input'
        WHERE 
        jt.jeditaskid IN ({jeditaskid_str})  and jd.type = 'input'
        GROUP BY 
        jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.taskname, jt.corecount
        """
    
        # Execute the combined query and return the resulting DataFrame
        df = pd.read_sql(query, con=conn)
        return df
    jeditaskids = [27766704, 27766716, 27766187, 27746332]
    df = fetch_task_param(jeditaskids, conn)
    print(df)
#    db_fetcher = DatabaseFetcher()
#    jeditaskids = [27766704, 27746332]  # Example task IDs
#    df = db_fetcher.fetch_task_param(jeditaskids)
#print(df)
