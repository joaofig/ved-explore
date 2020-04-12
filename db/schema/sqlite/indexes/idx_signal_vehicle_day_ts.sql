CREATE INDEX idx_signal_vehicle_day_ts ON signal (
    vehicle_id, day_num, time_stamp
);