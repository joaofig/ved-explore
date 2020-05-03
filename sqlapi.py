import sqlite3
import os
import json
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio


class SqlCache(object):

    def __init__(self, sql_dir='./db/sql'):
        self.sql_dir = sql_dir
        self.cache = dict()

    def get(self, key: str) -> str:
        if key not in self.cache:
            file_name = os.path.join(self.sql_dir, key + ".sql")
            if os.path.exists(file_name):
                with open(file_name) as sql_file:
                    sql = sql_file.read()
                    self.cache[key] = sql
        return self.cache[key]


class VedDb(object):

    def __init__(self, folder='./db'):
        self.db_folder = folder
        self.db_file_name = os.path.join(folder, 'ved.db')
        self.sql_cache = SqlCache()

        if not os.path.exists(self.db_file_name):
            self.create_schema()

    def connect(self):
        return sqlite3.connect(self.db_file_name, check_same_thread=False)

    def insert_vehicles(self, vehicles):
        conn = self.connect()
        cur = conn.cursor()
        sql = self.sql_cache.get("vehicle/insert")

        cur.executemany(sql, vehicles)

        conn.commit()
        cur.close()
        conn.close()

    def insert_signals(self, signals):
        conn = self.connect()
        cur = conn.cursor()
        sql = self.sql_cache.get("signal/insert")

        cur.executemany(sql, signals)

        conn.commit()
        cur.close()
        conn.close()

    def execute_sql(self, sql):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()

    def query_df(self, sql: str,
                 convert_none: bool = True) -> pd.DataFrame:
        conn = self.connect()
        df = sqlio.read_sql_query(sql, conn)
        if convert_none:
            df.fillna(value=np.nan, inplace=True)
        conn.close()
        return df

    def query(self, sql: str):
        conn = self.connect()
        cur = conn.cursor()
        result = list(cur.execute(sql))
        cur.close()
        conn.close()
        return result

    def head(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).head(rows)

    def tail(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).tail(rows)

    def generate_moves(self):
        sql = self.sql_cache.get("move/generate")
        self.execute_sql(sql)

    def create_schema(self, schema_dir='schema/sqlite'):
        schema_path = os.path.join(self.db_folder, schema_dir)
        schema_file_name = os.path.join(schema_path, "schema.json")

        with open(schema_file_name) as file:
            text = file.read()
        decoder = json.JSONDecoder()
        items = decoder.decode(text)

        item_sequence = items['sequence']

        for item_name in item_sequence:
            for file in items[item_name]:
                file_name = os.path.join(schema_path, file)
                with open(file_name) as sql_file:
                    sql = sql_file.read()
                    self.execute_sql(sql)


# Test code
def main():
    db = VedDb()


if __name__ == "__main__":
    main()
