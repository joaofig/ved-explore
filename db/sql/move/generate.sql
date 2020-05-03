insert into move (vehicle_id, day_num, ts_ini, ts_end)
select    tt.vehicle_id
,         tt.day_num
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
from (select distinct vehicle_id, day_num from signal) tt;
