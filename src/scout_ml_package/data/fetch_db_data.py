# # # data/fetch_db_data.py

import oracledb
import pandas as pd
import os
import configparser


class DatabaseFetcher:
    def __init__(self, db_config_name):
        self.db_config_name = self.db_config_name
        self.config = self.load_config()
        self.conn = self.get_db_connection()

    def load_config(self):
        # Load configuration
        config = configparser.ConfigParser()

        # Get the directory of the current script and construct path to config.ini
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # config_path = os.path.join(current_dir, '..', 'config.ini')  # Adjust path if necessary
        config_path = os.path.join(
            "/data/model-data/configs", "", "config.ini"
        )
        print(f"Config file path: {config_path}")
        print(f"Config file exists: {os.path.exists(config_path)}")

        # Read configuration file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config.read(config_path)

        if db_config_name not in config:
            raise KeyError("'database' section missing in config file")

        return config

    def get_db_connection(self, db_config_name):
        # Get database credentials
        database_user = self.config[self.db_config_name].get("user")
        database_password = self.config[self.db_config_name].get("password")
        dsn = self.config[self.db_config_name].get("dsn")

        try:
            # Establish connection with tcp_connect_timeout
            conn = oracledb.connect(
                user=database_user,
                password=database_password,
                dsn=dsn,
                tcp_connect_timeout=20,
            )  # 20 seconds timeout
            return conn

        except Exception as e:
            print(f"Failed to connect to the database: {e}")
            return None
            
    def reconnect_if_needed(self):
        if not self.conn or not self.conn.is_healthy():
            self.close_connection()
            self.conn = self.get_db_connection(self.db_config_name)
            if not self.conn:
                raise Exception("Failed to reconnect to the database.")

    def fetch_task_param(self, jeditaskids):
        self.reconnect_if_needed()

        if not isinstance(jeditaskids, list):
            jeditaskids = [jeditaskids]

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

    def write_data(self, data, table_name):
        self.reconnect_if_needed()
        try:
            # Assuming data is a pandas DataFrame
            data.to_sql(
                name=table_name,
                con=self.conn,
                if_exists='append',  # Change to 'replace' if you want to replace existing data
                index=False
            )
            self.conn.commit()
            print(f"Data successfully written to {table_name}")
        except Exception as e:
            print(f"Failed to write data to {table_name}: {e}")

    def get_connection(self):
        """Return the database connection."""
        self.reconnect_if_needed()
        return self.conn

    def close_connection(self):
        """Close the database connection and print a message."""
        if self.conn:
            self.conn.close()
            print("Database connection closed successfully.")
            self.conn = None
        else:
            print("No active database connection to close.")
    # def __init_subclass__(cls, db_config_name):
    #     cls.db_config_name = db_config_name


# # Example usage
# if __name__ == "__main__":
#     db_fetcher = DatabaseFetcher()
#     df = db_fetcher.fetch_task_param([27766704, 27766716])
#     print(df)
#     db_fetcher.close_connection()


# import oracledb
# import pandas as pd
# import configparser
# import os


# class DatabaseFetcher:
#     def __init__(self):
#         self.config = self.load_config()
#         self.conn = self.get_db_connection()

#     def load_config(self):
#         # Load configuration
#         config = configparser.ConfigParser()

#         # Get the directory of the current script and construct path to config.ini
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         config_path = os.path.join(current_dir, '..', 'config.ini')  # Adjust path if necessary

#         print(f"Config file path: {config_path}")
#         print(f"Config file exists: {os.path.exists(config_path)}")

#         # Read configuration file
#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"Config file not found at {config_path}")

#         config.read(config_path)

#         if 'database' not in config:
#             raise KeyError("'database' section missing in config file")

#         return config

#     def get_connection(self):
#         """Return the database connection."""
#         return self.conn

#     def close_connection(self):
#         """Close the database connection."""
#         if self.conn:
#             self.conn.close()
#             self.conn = None

#     def get_db_connection(self):
#         # Get database credentials
#         database_user = self.config['database'].get('user')
#         database_password = self.config['database'].get('password')
#         dsn = self.config['database'].get('dsn')

#         try:
#             # Establish connection with tcp_connect_timeout
#             conn = oracledb.connect(user=database_user, password=database_password, dsn=dsn,
#                                     tcp_connect_timeout=20)  # 20 seconds timeout
#             return conn

#         except Exception as e:
#             print(f"Failed to connect to the database: {e}")

#     def fetch_task_param(self, jeditaskids):
#         if not isinstance(jeditaskids, list):
#             jeditaskids = [jeditaskids]

#         # Create a string for SQL IN clause
#         jeditaskid_str = ', '.join(map(str, jeditaskids))

#         # Combined SQL query
#         query = f"""
#         SELECT
#             jt.jeditaskid,
#             jt.prodsourcelabel,
#             jt.processingtype,
#             jt.transhome,
#             jt.transpath,
#             jt.cputimeunit,
#             jt.corecount,
#             SUM(jd.NFILES) AS total_nfiles,
#             SUM(jd.NEVENTS) AS total_nevents,
#             COUNT(DISTINCT jd.DATASETNAME) AS distinct_datasetname_count
#         FROM
#             atlas_panda.jedi_tasks jt
#         LEFT JOIN
#             atlas_panda.jedi_datasets jd ON jt.jeditaskid = jd.jeditaskid AND jd.TYPE = 'input'
#         WHERE
#             jt.jeditaskid IN ({jeditaskid_str}) AND jd.type = 'input'
#         GROUP BY
#             jt.jeditaskid, jt.prodsourcelabel, jt.processingtype, jt.transhome, jt.transpath, jt.cputimeunit, jt.corecount
#         """

#         # Execute the combined query and return the resulting DataFrame
#         df = pd.read_sql(query, con=self.conn)
#         return df
