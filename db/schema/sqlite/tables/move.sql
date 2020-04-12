CREATE TABLE move (
    move_id         INTEGER PRIMARY KEY ASC,
    day_num         FLOAT NOT NULL,
    vehicle_id      INT NOT NULL,
    ts_ini          INT NOT NULL,
    ts_end          INT NOT NULL
);