import oracledb
import pandas as pd

import sqlite3

print(sqlite3.sqlite_version)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
database_user = "atlas_pandaredwood_r"
database_password = "N4FFt*4n0McVj+"
dsn = "adcr-s.cern.ch:10121/adcr_panda.cern.ch"
# conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn)


# Establish a connection to the Oracle database
try:
    conn = oracledb.connect(
        user=database_user, password=database_password, dsn=dsn
    )
    # conn = oracledb.connect(user='your_username', password='your_password', dsn='your_dsn')
except oracledb.DatabaseError as e:
    print("There was a problem connecting to the database:", e)
    conn = None  # Set conn to None if the connection failed


def fetch_task_param(jeditaskids, conn):
    if not isinstance(jeditaskids, list):
        jeditaskids = [jeditaskids]

    # Create a string for SQL IN clause
    jeditaskid_str = ", ".join(map(str, jeditaskids))

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
        jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.corecount
    """

    # Execute the combined query and return the resulting DataFrame
    df = pd.read_sql(query, con=conn)
    return df


jeditaskids = [27766704, 27766716, 27766187, 27746332]

if conn:  # Check if connection is established
    df = fetch_task_param(jeditaskids, conn)
    print(df)
else:
    print("Connection is not established. Cannot fetch data.")


if conn:
    conn.close()
