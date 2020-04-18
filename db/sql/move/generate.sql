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
