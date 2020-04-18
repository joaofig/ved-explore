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
