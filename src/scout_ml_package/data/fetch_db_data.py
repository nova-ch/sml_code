# data/data_fetcher.py

import oracledb
import pandas as pd
import configparser
import os


def get_db_connection():
    # Load configuration
    config = configparser.ConfigParser()

    # Get the directory of the current script and construct path to config.ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir,'..', 'config.ini')  # Adjust path if necessary

    print(f"Config file path: {config_path}")
    print(f"Config file exists: {os.path.exists(config_path)}")

    # Read configuration file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config.read(config_path)

    if 'database' not in config:
        raise KeyError("'database' section missing in config file")

    # Get database credentials
    database_user = config['database'].get('user')
    database_password = config['database'].get('password')
    dsn = config['database'].get('dsn')

    # if not all([database_user, database_password, dsn]):
    #     raise KeyError("Missing required database credentials in [database] section")
    #
    # # Establish connection
    # conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn)
    # Establish connection with a timeout
    try:
        # Establish connection with tcp_connect_timeout
        conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn,
                                tcp_connect_timeout=20)  # 20 seconds timeout
        return conn

    except Exception as e:
        print(f"Failed to connect to the database: {e}")


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
        jt.jeditaskid IN ({jeditaskid_str}) AND jd.type = 'input'
    GROUP BY 
        jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.taskname, jt.corecount
    """

    # Execute the combined query and return the resulting DataFrame
    df = pd.read_sql(query, con=conn)
    return df


