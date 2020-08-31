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


class BaseDb(object):

    def __init__(self, folder='./db', file_name='database.db'):
        self.db_folder = folder
        self.db_file_name = os.path.join(folder, file_name)
        self.sql_cache = SqlCache()

    def connect(self):
        return sqlite3.connect(self.db_file_name, check_same_thread=False)

    def execute_sql(self, sql):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()

    def query_df(self, sql: str, parameters=None,
                 convert_none: bool = True) -> pd.DataFrame:
        conn = self.connect()
        df = sqlio.read_sql_query(sql, conn, params=parameters)
        if convert_none:
            df.fillna(value=np.nan, inplace=True)
        conn.close()
        return df

    def query(self, sql, parameters=[]):
        conn = self.connect()
        cur = conn.cursor()
        result = list(cur.execute(sql, parameters))
        cur.close()
        conn.close()
        return result

    def query_scalar(self, sql, parameters=[]):
        res = self.query(sql, parameters)
        return res[0][0]

    def head(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).head(rows)

    def tail(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).tail(rows)

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

    def table_has_column(self, table, column):
        sql = "PRAGMA table_info ('{}')".format(table)
        lst = self.query(sql)
        for col in lst:
            if col[1] == column:
                return col
        return None

    def table_exists(self, table_name):
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = set([table[0] for table in self.query(sql)])
        return table_name in tables

    def insert_list(self, sql_cache_key, values):
        conn = self.connect()
        cur = conn.cursor()
        sql = self.sql_cache.get(sql_cache_key)

        cur.executemany(sql, values)

        conn.commit()
        cur.close()
        conn.close()


class VedDb(BaseDb):

    def __init__(self, folder='./db'):
        super().__init__(folder=folder, file_name='ved.db')

        if not os.path.exists(self.db_file_name):
            self.create_schema()

    def insert_vehicles(self, vehicles):
        self.insert_list("vehicle/insert", vehicles)

    def insert_signals(self, signals):
        self.insert_list("signal/insert", signals)

    def insert_cluster_points(self, cluster_points):
        self.insert_list("cluster_point/insert", cluster_points)

    def generate_moves(self):
        sql = self.sql_cache.get("move/generate")
        self.execute_sql(sql)
        
    def update_move_clusters(self, clusters):
        conn = self.connect()
        cur = conn.cursor()
        sql = self.sql_cache.get("move/update_clusters")

        cur.executemany(sql, clusters)

        conn.commit()
        cur.close()
        conn.close()

    def update_move_h3s(self, h3s):
        conn = self.connect()
        cur = conn.cursor()
        sql = self.sql_cache.get("move/update_h3s")

        cur.executemany(sql, h3s)

        conn.commit()
        cur.close()
        conn.close()


# Test code
def main():
    db = VedDb()


if __name__ == "__main__":
    main()
