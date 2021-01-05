import os
import pandas as pd

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from db.api import VedDb, TileDb
from pyquadkey2 import quadkey
from pyquadkey2.quadkey import tilesystem
from scipy.stats import multivariate_normal
from pathlib import Path


def qk_str_toint(qk_str):
    qk_int = 0
    for c in qk_str:
        qk_int = (qk_int << 2) + int(c)
    return qk_int


def qk_toint(qk):
    return qk_str_toint(str(qk))


def get_unique_locations(db):
    """
    Gets an iterator to the unique locations (lat, lon)
    :param db: Source database (VedDb)
    :return: Iterator to the unique locations
    """
    sql = "select distinct latitude, longitude from signal"
    locations = db.query_iterator(sql)
    return locations


def update_density(f0, f1, qk):
    tile = qk.to_tile()
    qk_x = tile[0][0]
    qk_y = tile[0][1]
    zoom = tile[1]
    n2 = multivariate_normal(cov=1, mean=[0.0, 0.0])
    for x in range(-4, 5):
        for y in range(-4, 5):
            p = n2.pdf([x, y])
            if p > 1e-5:
                qk_xy = quadkey.from_str(tilesystem.tile_to_quadkey((qk_x + x, qk_y + y), zoom))
                if qk.key[:14] == qk_xy.key[:14]:
                    f0.write('"{0}",{1}\n'.format(qk_xy.key, p))
                else:
                    f1.write('"{0}","{1}",{2}\n'.format(qk_xy.key[:14], qk_xy.key, p))


def write_tile(tile):
    root = tile[0]
    quadkeys = tile[1]

    base_folder = "./tiles/tmp/"
    Path(base_folder).mkdir(parents=True, exist_ok=True)

    f0 = open(os.path.join(base_folder, root + ".csv"), "w", buffering=16 * 1024 * 1024)
    f1 = open(os.path.join(base_folder, "n_" + root + ".csv"), "w")

    f0.write('"quadkey","density"\n')
    f1.write('"root","quadkey","density"\n')

    for qk in quadkeys:
        update_density(f0, f1, qk)

    f1.close()
    f0.close()


def process_csv_files(tile_db, base_folder="tiles/tmp"):
    """
    Processes the generated CSV files by doing a group and sum.
    At the end, the function inserts the data for level 26
    :param tile_db: Tile database
    :param base_folder: CSV folder
    :return: Nothing
    """
    neighbor_files = [f for f in os.listdir(base_folder) if f.startswith("n_")]
    tile_files = [f for f in os.listdir(base_folder) if not f.startswith("n_")]

    n_df = pd.concat((pd.read_csv(os.path.join(base_folder, f)) for f in tqdm(neighbor_files)))

    for tf in tqdm(tile_files):
        df = pd.read_csv(os.path.join(base_folder, tf))
        tile = tf[:14]

        df = pd.concat([df, n_df.loc[n_df["root"] == tile, ["quadkey", "density"]]])
        dens_df = df.groupby("quadkey")["density"].sum().to_frame().reset_index()

        level_list = []
        qk_arr = dens_df["quadkey"].to_numpy()
        de_arr = dens_df["density"].to_numpy()

        for i in range(qk_arr.shape[0]):
            qk_int = qk_str_toint(qk_arr[i])
            level_list.append((qk_int, qk_int >> 16, de_arr[i]))

        tile_db.insert_qk_intensities(26, level_list)


def generate_levels(tile_db):
    """
    Generates levels 25 to 8 using level 26 as the source
    :param tile_db: Tile database
    :return: Nothing
    """
    sql = """
    insert into l{0:02} (qk, tile, intensity)
    select   qk / 4 as qk, 
             tile / 4 as tile, 
             sum(intensity) as intensity 
    from     l{1:02} 
    group by qk / 4;"""

    for level in tqdm(range(26, 8, -1)):
        tile_db.execute_sql(sql.format(level - 1, level))


def calculate_level_ranges(tile_db):
    """
    Calculates the level intensity / density ranges
    :param tile_db: Tile database
    :return: Nothing
    """
    sql = """
    insert into level_range (level_num, level_min, level_max)
        select {0}            as level_num
        ,      min(intensity) as level_min
        ,      max(intensity) as level_max
        from   l{1:02} 
    """

    for level in tqdm(range(26, 7, -1)):
        tile_db.execute_sql(sql.format(level, level))


def get_tile_list(ved_db):
    """
    Retrieves the tile list corresponding to the unique
    locations in the VED database
    :param ved_db: The VED database
    :return: List of tiles encoded as quadkeys
    """
    tiles = dict()

    with get_unique_locations(ved_db) as locations:
        for p in tqdm(locations):
            # qk is a Quadkey object
            qk = quadkey.from_geo((p[0], p[1]), level=26)
            tile = qk.key[:14]

            if tile in tiles:
                tiles[tile].append(qk)
            else:
                tiles[tile] = [qk]

    tile_list = [(k, v) for k, v in tiles.items()]
    return tile_list


def main():
    tile_db = TileDb(folder="./db")
    ved_db = VedDb(folder="./db")

    print("Retrieving tile list")
    tile_list = get_tile_list(ved_db)

    print("Process locations")
    process_map(write_tile, tile_list, max_workers=12)

    print("Process CSV files")
    process_csv_files(tile_db)

    print("Generate level data")
    generate_levels(tile_db)

    print("Calculate level ranges")
    calculate_level_ranges(tile_db)


if __name__ == "__main__":
    main()
