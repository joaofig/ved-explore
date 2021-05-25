import os
import png
import numpy as np

from flask import Flask, send_from_directory, make_response
from pyquadkey2.quadkey import tilesystem
from pyquadkey2 import quadkey
from numba import jit
from colour import Color
from pathlib import Path
from db.api import TileDb, TraceDb

MAX_COLORS = 256


app = Flask(__name__)


def qk_int_to_str(qk_int, z):
    qk_str = ""
    for i in range(z):
        qk_str = "0123"[qk_int % 4] + qk_str
        qk_int = qk_int >> 2
    return qk_str


def qk_str_to_int(qk_str):
    qk_int = 0
    for c in qk_str:
        qk_int = (qk_int << 2) + int(c)
    return qk_int


@jit(nopython=True)
def make_rgba(r, g, b, a):
    """
    Creates a 256x256 tile using the given red, green, blue and alpha channels
    :param r: Red channel value
    :param g: Green channel value
    :param b: Blue channel value
    :param a: Alpha channel value
    :return: A 256x256x4 integer NumPy array initialized with the channel data
    """
    pic = np.zeros((256, 256, 4), dtype=np.uint8)
    pic[:, :, 0] = np.uint8(r)
    pic[:, :, 1] = np.uint8(g)
    pic[:, :, 2] = np.uint8(b)
    pic[:, :, 3] = np.uint8(a)
    return pic


def create_color_list():
    """
    Creates the color gradient list
    :return: Color gradient NumPy array (MAX_COLORS, 4)
    """
    ini = Color("cyan")
    end = Color("#ff0000")
    color_range = list(ini.range_to(end, MAX_COLORS))
    colors = np.array([np.array([int(c.red * 255),
                                 int(c.green * 255),
                                 int(c.blue * 255), 127]) for c in color_range])
    return colors


def get_tile_quadkeys(db, qk):
    sql = "select qk, intensity from l{0:02} where tile=?".format(qk.level + 8)
    return db.query(sql, [qk_str_to_int(str(qk))])


def save_tile(pic, file_name):
    """
    Saves a tile to a PNG-formatted file
    :param pic: The NumPy array containing the tile data
    :param file_name: Target file name
    :return: Nothing
    """
    pic2d = np.reshape(pic, (pic.shape[0], pic.shape[1] * pic.shape[2]))
    with open(file_name, "wb") as file:
        writer = png.Writer(width=256, height=256, greyscale=False,
                            alpha=True, bitdepth=8)
        writer.write(file, pic2d)


def make_tile_response(file_name):
    """
    Creates the response object from the tile file name
    :param file_name: Tile file name
    :return: Response
    """
    with open(file_name, "rb") as file:
        image = file.read()
        response = make_response(image)
        response.headers.set('Content-Type', 'image/png')
        return response


def get_default_file_name():
    file_name = "./tiles/default.png"
    if not os.path.isfile(file_name):
        Path("./tiles").mkdir(parents=True, exist_ok=True)

        pic = make_rgba(0, 0, 0, 0)
        save_tile(pic, file_name)
    return file_name


def get_tile_pixels(qk, tile_qks):
    pixel = qk.to_pixel()
    x0 = pixel[0]
    y0 = pixel[1]

    pixels = []
    for row in tile_qks:
        qki = quadkey.from_str(qk_int_to_str(row[0], qk.level + 8))
        pi = qki.to_pixel()
        xi = pi[0]
        yi = pi[1]
        pixels.append(np.array([xi // 256 - x0, yi // 256 - y0, row[1]]))
    return pixels


@jit(nopython=True)
def paint_tile(pixels, colors, rng):
    """
    Paints a tile using NumPy
    :param pixels: List of pixels to paint [(x, y, intensity)].
    :param colors: List of RGB colors.
    :param rng: Tuple containing the zoom level intensity range (min, max).
    :return: A NumPy array representing the tile bitmap.
    """
    pic = make_rgba(0, 0, 0, 0)     # Make a fully transparent tile
    for x, y, c in pixels:
        if rng[0] != rng[1]:
            ic = int(((c - rng[0]) * MAX_COLORS) // (rng[1] - rng[0]))
            pic[int(y), int(x)] = colors[ic]
        else:
            pic[int(y), int(x)] = colors[-1]
    return pic


def get_generic_tile(x, y, zoom, db_file_name, folder):
    """
    Retrieves or generates a generic tile
    :param x: Tile X coordinate
    :param y: Tile Y coordinate
    :param zoom: Zoom level
    :param db_file_name: SQLite database file name with the level tables
    :param folder: Cache folder path
    :return: A REST response with the tile PNG file
    """
    if zoom < 1 or zoom > 18:
        file_name = get_default_file_name()
    else:
        qk = quadkey.from_str(tilesystem.tile_to_quadkey((x, y), zoom))
        file_name = "./tiles/{0}/{1}/{2}.png".format(folder, zoom, str(qk))

        if not os.path.isfile(file_name):
            db = TileDb(file_name=db_file_name)
            tile_qks = get_tile_quadkeys(db, qk)
            if len(tile_qks) == 0:
                file_name = get_default_file_name()
            else:
                tile_pixels = get_tile_pixels(qk, tile_qks)
                r = db.get_level_range(qk.level + 8)
                pic = paint_tile(tile_pixels, create_color_list(), r)
                Path("./tiles/{0}/{1}".format(folder, zoom)).mkdir(parents=True, exist_ok=True)
                save_tile(pic, file_name)
    return make_tile_response(file_name)


@app.route("/density/<int:x>/<int:y>/<int:z>/", methods=["GET"])
def get_density_tile(x: int, y: int, z: int):
    return get_generic_tile(x, y, z, "tile.db", "density")


@app.route("/trace/<int:x>/<int:y>/<int:z>/", methods=["GET"])
def get_trace_tile(x, y, z):
    return get_generic_tile(x, y, z, "trace.db", "trace")


@app.route("/hello/", methods=["GET"])
def hello():
    return os.path.dirname(app.instance_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2310, threaded=True)
