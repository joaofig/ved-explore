import numpy as np
import math


def vec_haversine(lat1: np.ndarray,
                  lon1: np.ndarray,
                  lat2: np.ndarray,
                  lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance calculation
    :param lat1: Array of initial latitudes in degrees
    :param lon1: Array of initial longitudes in degrees
    :param lat2: Array of destination latitudes in degrees
    :param lon2: Array of destination longitudes in degrees
    :return: Array of distances in meters
    """
    earth_radius = 6378137.0

    rad_lat1 = np.radians(lat1)
    rad_lon1 = np.radians(lon1)
    rad_lat2 = np.radians(lat2)
    rad_lon2 = np.radians(lon2)

    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1

    a = np.sin(d_lat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) \
        * np.sin(d_lon/2.0)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


def num_haversine(lat1: float,
                  lon1: float,
                  lat2: float,
                  lon2: float) -> float:
    """
    Haversine distance calculation
    :param lat1: Initial latitude in degrees
    :param lon1: Initial longitude in degrees
    :param lat2: Destination latitude in degrees
    :param lon2: Destination longitude in degrees
    :return: Distances in meters
    """
    earth_radius = 6378137.0

    rad_lat1 = math.radians(lat1)
    rad_lon1 = math.radians(lon1)
    rad_lat2 = math.radians(lat2)
    rad_lon2 = math.radians(lon2)

    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1

    a = math.sin(d_lat/2.0)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) \
        * math.sin(d_lon/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


def delta_location(lat: float,
                   lon: float,
                   bearing: float,
                   meters: float) -> (float, float):
    """
    Calculates a destination location from a starting location, a bearing and a
    distance in meters.
    :param lat: Start latitude
    :param lon: Start longitude
    :param bearing: Bearing (North is zero degrees, measured clockwise)
    :param meters: Distance to displace from the starting point
    :return: Tuple with the new latitude and longitude
    """
    delta = meters / 6378137.0
    theta = math.radians(bearing)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat_r2 = math.asin(math.sin(lat_r) * math.cos(delta) + math.cos(lat_r) *
                       math.sin(delta) * math.cos(theta))
    lon_r2 = lon_r + math.atan2(math.sin(theta) * math.sin(delta) *
                                math.cos(lat_r),
                                math.cos(delta) - math.sin(lat_r) *
                                math.sin(lat_r2))
    return math.degrees(lat_r2), math.degrees(lon_r2)


def x_meters_to_degrees(meters: float,
                        lat: float,
                        lon: float) -> float:
    """
    Converts a horizontal distance in meters to an angle in degrees.
    :param meters: Distance to convert
    :param lat: Latitude of reference location
    :param lon: Longitude of reference location
    :return: Horizontal angle in degrees
    """
    _, lon2 = delta_location(lat, lon, 90, meters)
    return abs(lon - lon2)


def y_meters_to_degrees(meters: float,
                        lat: float,
                        lon: float) -> float:
    """
    Converts a vertical distance in meters to an angle in degrees.
    :param meters: Distance to convert
    :param lat: Latitude of reference location
    :param lon: Longitude of reference location
    :return: Vertical angle in degrees
    """
    lat2, _ = delta_location(lat, lon, 0, meters)
    return abs(lat - lat2)
