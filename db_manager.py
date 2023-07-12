import logging
from sqlalchemy.exc import DBAPIError, StatementError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table
from abc import ABC, abstractmethod
from retry import retry
import os
import re
import csv
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager(ABC):
    def __init__(self, username=None, password=None, host=None, database=None):
        self.username = username
        self.password = password
        self.host = host
        self.database = database
        self.engine = None
        self.session = None

    @abstractmethod
    def connect(self):
        pass

    @retry(DBAPIError, tries=3, delay=2)
    def execute(self, query):
        try:
            if not self.session:
                raise Exception("Database not connected")
            logger.info(f"Executing query: {query}")
            self.session.execute(query)
            self.session.commit()
            return True
        except StatementError as e:
            logger.error(f"Error executing query: {e}")
            return None

    def import_data(self, file_path, table_name, delimiter=",", has_header=True):
        try:
            with open(file_path, 'r') as data_file:
                data_reader = csv.reader(data_file, delimiter=delimiter)
                headers = next(data_reader) if has_header else None

                metadata = MetaData(bind=self.engine)
                table = Table(table_name, metadata, autoload_with=self.engine)

                for row in data_reader:
                    insert_dict = {column.name: value for column,
                                   value in zip(table.columns, row)}
                    self.engine.execute(table.insert().values(insert_dict))

            self.logger.info(f"Data imported successfully into {table_name}")
        except SQLAlchemyError as e:
            self.logger.error(
                f"Error while importing data to {table_name}: {str(e)}")
            return False
        return True

    def extract_data(self, table=None, query=None, output_file=None, delimiter=',', header=True, chunk_size=1000):
        if not self.session:
            raise Exception("Database not connected")

        if query is None and table is not None:
            query = f'SELECT * FROM {table}'
        elif query is None and table is None:
            raise ValueError(
                "Both query and table cannot be None. Please provide at least one.")

        if output_file is None:
            raise ValueError(
                "output_file cannot be None. Please provide a valid path.")

        logger.info(f"Executing query: {query}")
        try:
            result_proxy = self.session.execute(query)
            if header:
                columns = result_proxy.keys()
            try:
                with open(output_file, 'w') as f:
                    writer = csv.writer(f, delimiter=delimiter)
                    if header:
                        writer.writerow(columns)
                    while True:
                        chunk = result_proxy.fetchmany(chunk_size)
                        if not chunk:
                            break
                        for row in chunk:
                            writer.writerow(row)

                logger.info("Data extracted successfully")
            except IOError as e:
                logger.error(f"I/O error({e.errno}): {e.strerror}")
        except Exception as e:
            logger.error(f"Error executing query: {e}")

    def close(self):
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
        except DBAPIError as e:
            logger.error(f"Error while closing the database connection: {e}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MSSQLManager(DatabaseManager):
    def __init__(self, username=None, password=None, host=None, database=None, trusted_connection=False):
        super().__init__(username, password, host, database)
        self.trusted_connection = trusted_connection

    def connect(self):
        try:
            if self.trusted_connection:
                connection_string = f"mssql+pyodbc://{self.host}/{self.database}?driver=ODBC+Driver+17+for+SQL+Server;Trusted_Connection=yes"
            else:
                connection_string = f"mssql+pyodbc://{self.username}:{self.password}@{self.host}/{self.database}?driver=ODBC+Driver+17+for+SQL+Server"
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        except DBAPIError as e:
            logger.error(f"Error connecting to the database: {e}")

    def bulk_import_data(self, file_path, table_name, delimiter=",", has_header=True):
        try:
            start_time = time.time()
            query = f"""
                BULK INSERT {table_name}
                FROM '{file_path}'
                WITH (
                    FIELDTERMINATOR = '{delimiter}',
                    ROWTERMINATOR = '\\n',
                    FIRSTROW = {2 if has_header else 1},
                    TABLOCK
                )
            """
            self.execute(query)
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            elapsed_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds)

            logger.info(
                f"{file_path} imported successfully into {table_name} in {elapsed_time_str} (HH:MM:SS)")
        except SQLAlchemyError as e:
            logger.error(
                f"Error while importing data to {table_name}: {str(e)}")
            return False
        return True


class PostgresManager(DatabaseManager):
    @retry(DBAPIError, tries=3, delay=2)
    def connect(self):
        try:
            connection_string = f"postgresql://{self.username}:{self.password}@{self.host}/{self.database}"
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        except DBAPIError as e:
            # logger.error(f"Error connecting to the database: {e}")
            raise


class VerticaManager(DatabaseManager):
    @retry(DBAPIError, tries=3, delay=2)
    def connect(self):
        try:
            connection_string = f"vertica+pyodbc://{self.username}:{self.password}@{self.host}/{self.database}"
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        except DBAPIError as e:
            logger.error(f"Error connecting to the database: {e}")
            raise
