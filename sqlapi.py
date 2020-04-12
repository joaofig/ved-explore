import sqlite3
import os
import json
import pandas.io.sql as sqlio


class VedDb(object):

    def __init__(self, folder='./db'):
        self.db_folder = folder
        self.db_file_name = os.path.join(folder, 'ved.db')

        if not os.path.exists(self.db_file_name):
            self.create_schema()

    def connect(self):
        return sqlite3.connect(self.db_file_name, check_same_thread=False)

    def insert_vehicles(self, vehicles):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
        INSERT INTO vehicle (
            vehicle_id,
            vehicle_type,
            vehicle_class,
            engine,
            transmission,
            drive_wheels,
            weight
            )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        ''', vehicles)

        conn.commit()
        cur.close()
        conn.close()

    def insert_signals(self, signals):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
        INSERT INTO signal (
            day_num,
            vehicle_id,
            trip_id,
            time_stamp,
            latitude,
            longitude,
            speed,
            maf,
            rpm,
            abs_load,
            oat,
            fuel_rate,
            ac_power_kw,
            ac_power_w,
            heater_power_w,
            hv_bat_current,
            hv_bat_soc,
            hv_bat_volt,
            st_ftb_1,
            st_ftb_2,
            lt_ftb_1,
            lt_ftb_2
            )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?
        );
        ''', signals)

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

    def generate_moves(self):
        sql = '''
insert into move (day_num, vehicle_id, ts_ini, ts_end)
select    tt.day_num
,         tt.vehicle_id
,         ( select min(s1.time_stamp) 
            from signal s1 
            where s1.day_num = tt.day_num and 
                  s1.vehicle_id = tt.vehicle_id
            ) as ts_ini
,         ( select max(s2.time_stamp) 
            from signal s2 
            where s2.day_num = tt.day_num and 
                  s2.vehicle_id = tt.vehicle_id
            ) as ts_end
from (select distinct day_num, vehicle_id from signal) tt;
        '''
        self.execute_sql(sql)

    def create_schema(self, schema_dir='schema/sqlite'):
        schema_path = os.path.join(self.db_folder, schema_dir)
        schema_file_name = os.path.join(schema_path, "schema.json")

        with open(schema_file_name) as file:
            text = file.read()
        decoder = json.JSONDecoder()
        items = decoder.decode(text)

        item_names = ['tables', 'indexes']

        for item_name in item_names:
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
