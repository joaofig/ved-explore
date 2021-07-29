CREATE TABLE cluster_point (
    pt_id           INTEGER PRIMARY KEY ASC,
    cluster_id      INT NOT NULL,
    latitude        FLOAT NOT NULL,
    longitude       FLOAT NOT NULL,
    h3              TEXT
);