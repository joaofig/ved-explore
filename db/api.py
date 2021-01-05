import sqlite3
import os
import json
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import contextlib


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

    def execute_sql(self, sql, parameters=[], many=False):
        conn = self.connect()
        cur = conn.cursor()
        if not many:
            cur.execute(sql, parameters)
        else:
            cur.executemany(sql, parameters)
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

    @contextlib.contextmanager
    def query_iterator(self, sql, parameters=[]):
        conn = self.connect()
        cur = conn.cursor()
        yield cur.execute(sql, parameters)
        cur.close()
        conn.close()

    def query_scalar(self, sql, parameters=[]):
        res = self.query(sql, parameters)
        return res[0][0]

    def head(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).head(rows)

    def tail(self, sql: str, rows: int = 5) -> pd.DataFrame:
        return self.query_df(sql).tail(rows)

    def create_schema(self, schema_dir):
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
            self.create_schema(schema_dir='schema/ved')

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


class TileDb(BaseDb):

    def __init__(self, folder='./db', file_name='tile.db'):
        super().__init__(folder=folder, file_name=file_name)

        if not os.path.exists(self.db_file_name):
            self.create_schema(schema_dir='schema/tile')

            for i in range(8, 27):
                sql = """
                CREATE TABLE l{0:02} (
                    qk          INTEGER PRIMARY KEY,
                    tile        INTEGER NOT NULL,
                    intensity   FLOAT NOT NULL
                );
                """.format(i)
                self.execute_sql(sql)

                sql = """
                CREATE INDEX idx_l{0:02}_tile ON l{0:02} (
                    tile
                );
                """.format(i)
                self.execute_sql(sql)

    def insert_qk_intensities(self, level, pair_list):
        sql = "insert into l{0:02} (qk, tile, intensity) values (?, ?, ?)".format(level)
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany(sql, pair_list)

        conn.commit()
        cur.close()
        conn.close()

    def insert_level_ranges(self, level_ranges):
        sql = "insert into level_range (level_num, level_min, level_max) values (?, ?, ?)"
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany(sql, level_ranges)

        conn.commit()
        cur.close()
        conn.close()

    def get_level_range(self, lvl):
        sql = "select level_min, level_max from level_range where level_num=?"
        conn = self.connect()
        cur = conn.cursor()

        r = list(cur.execute(sql, [lvl]))[0]

        conn.commit()
        cur.close()
        conn.close()
        return r


class TraceDb(TileDb):

    def __init__(self, folder='./db', file_name='trace.db'):
        super().__init__(folder=folder, file_name=file_name)


# Test code
def main():
    vedDb = VedDb()
    tileDb = TileDb()


if __name__ == "__main__":
    main()
