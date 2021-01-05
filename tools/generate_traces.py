from collections import Counter

import numpy as np

from tqdm.auto import tqdm
from db.api import VedDb, TraceDb
from numba import jit
from pyquadkey2 import quadkey


def qk_int_tostr(qk_int, z):
    qk_str = ""
    for i in range(z):
        qk_str = "0123"[qk_int % 4] + qk_str
        qk_int = qk_int >> 2
    return qk_str


def qk_str_toint(qk_str):
    qk_int = 0
    for c in qk_str:
        qk_int = (qk_int << 2) + int(c)
    return qk_int


def qk_toint(qk):
    return qk_str_toint(str(qk))


def get_master_tile(qk, z):
    qk_str = qk_int_tostr(qk, z)
    tile = quadkey.from_str(qk_str[:len(qk_str) - 8])
    return tile


@jit(nopython=True)
def bresenham_pairs(x0: int, y0: int,
                    x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates

    Parameters
    ----------
    x0 : int
        Origin x value
    y0 : int
        Origin y value
    x1 : int
        Target x value
    y1 : int
        Target y value

    Returns
    -------
    np.ndarray
        Array with the diagonal coordinates
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


def create_pixel_table(db):
    if not db.table_exists("pixel"):
        sql = """
        CREATE TABLE pixel (
            pixel_id    INTEGER PRIMARY KEY ASC,
            qk          INTEGER NOT NULL,
            intensity   FLOAT NOT NUL
        );
        """
        db.execute_sql(sql)

        # sql = "CREATE INDEX idx_pixel_qk ON pixel (qk);"
        # db.execute_sql(sql)


def get_moves(db):
    sql = "select vehicle_id, day_num from move"
    moves = db.query(sql)
    return moves


def get_move_points(db, move):
    sql = """
    select   latitude, longitude 
    from     signal 
    where    vehicle_id = ? and day_num = ?
    order by time_stamp;
    """
    points = db.query(sql, move)
    return points


def get_unique_points(points):
    unique_pts = []
    last_pt = None
    for pt in points:
        if last_pt is None or last_pt != pt:
            unique_pts.append(pt)
        last_pt = pt
    return unique_pts


def add_pixel(db, px, intensity):
    sql = "select n from l26 where qk=?"
    res = db.query(sql, parameters=[px])
    if len(res) == 0:
        sql = "insert into l26 (qk, tile, n) values (?, ?, ?)"
        db.execute_sql(sql, [px, px // 256, intensity])
    else:
        sql = "update l26 set n = ? where qk = ?"
        db.execute_sql(sql, [res[0] + intensity, px])


def get_level_list(d, z):
    l = [(qk, qk_toint(get_master_tile(qk, z)), d[qk]) for qk in d.keys()]
    return l


def get_parent_level(d):
    parents = dict()
    for qk in d.keys():
        p = qk // 4
        if p in parents:
            parents[p] += d[qk]
        else:
            parents[p] = d[qk]
    return parents


def insert_all_levels(db, counter):
    level = counter
    level_ranges = dict()
    for z in tqdm(range(26, 8, -1)):
        level_ranges[z] = (min(level.values()), max(level.values()))
        db.insert_qk_intensities(z, get_level_list(level, z))
        level = get_parent_level(level)
    return level_ranges


def main():
    trace_db = TraceDb(folder="../db")
    ved_db = VedDb(folder="../db")

    counter = Counter()

    create_pixel_table(trace_db)
    moves = get_moves(ved_db)
    for move in tqdm(moves):
        points = get_unique_points(get_move_points(ved_db, move))

        tiles = [quadkey.from_geo((p[0], p[1]), level=26).to_tile() for p in points]

        for i in range(len(tiles) - 1):
            x0, y0 = tiles[i][0]
            x1, y1 = tiles[i+1][0]
            line = bresenham_pairs(x0, y0, x1, y1)
            pixels = [qk_str_toint(quadkey.from_tile((p[0], p[1]), 26).key) for p in line]
            counter.update(pixels)

    level_ranges = insert_all_levels(trace_db, counter)

    trace_db.insert_level_ranges([(k, v[0], v[1]) for k, v in level_ranges.items()])


if __name__ == "__main__":
    main()
