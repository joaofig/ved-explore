CREATE TABLE move (
    move_id         INTEGER PRIMARY KEY ASC,
    vehicle_id      INT NOT NULL,
    day_num         FLOAT NOT NULL,
    ts_ini          INT NOT NULL,
    ts_end          INT NOT NULL,
    cluster_ini     INT NOT NULL DEFAULT -1,
    cluster_end     INT NOT NULL DEFAULT -1
);