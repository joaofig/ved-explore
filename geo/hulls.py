import numpy as np
import math
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import transform
from functools import partial
import pyproj


def haversine_distance(loc_ini, loc_end):
    lon1, lat1, lon2, lat2 = map(np.radians,
                                 [loc_ini[0], loc_ini[1],
                                  loc_end[:, 0], loc_end[:, 1]])

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = np.square(np.sin(delta_lat / 2.0)) + \
        np.cos(lat1) * np.cos(lat2) * np.square(np.sin(delta_lon / 2.0))

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = 6378137.0 * c
    return meters


def get_next_k(k: int) -> int:
    return k * k


class ConcaveHull(object):

    def __init__(self, points):
        if isinstance(points, np.core.ndarray):
            self.data_set = np.copy(points)
        elif isinstance(points, list):
            self.data_set = np.copy(np.array(points))
        else:
            raise ValueError('Please provide an [N,2] numpy array or a list of lists.')

        # Clean up duplicates
        # self.data_set = np.unique(self.data_set, axis=0)

        # Create the initial index
        self.indices = np.ones(self.data_set.shape[0], dtype=bool)

    @staticmethod
    def buffer_in_meters(hull, meters):
        proj_meters = pyproj.Proj('epsg:3857')
        proj_latlng = pyproj.Proj('epsg:4326')

        project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
        project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

        hull_meters = transform(project_to_meters, hull)

        buffer_meters = hull_meters.buffer(meters)
        buffer_latlng = transform(project_to_latlng, buffer_meters)
        return buffer_latlng

    @staticmethod
    def get_lowest_latitude_index(points):
        indices = np.argsort(points[:, 1])
        return indices[0]

    def get_k_nearest(self, ix, k):
        """
        Calculates the k nearest point indices to the point indexed by ix
        :param ix: Index of the starting point
        :param k: Number of neighbors to consider
        :return: Array of indices into the data set array
        """
        ixs = self.indices.copy()
        
        ixs[ix] = False

        base_indices = np.arange(len(ixs))[ixs]
        distances = haversine_distance(self.data_set[ix, :], self.data_set[ixs, :])
        sorted_indices = np.argsort(distances)

        # kk = min(k, len(sorted_indices))
        k_nearest = sorted_indices[:k]
        return base_indices[k_nearest]

    def calculate_bearings(self, ix, ixs, ref_bearing=0.0):
        """
        Calculates the bearings from a source point to a set of target points.
        :param ix: Index to the source point in the data set
        :param ixs: Indexes to the target points in the data set
        :param ref_bearing: Reference bearing measured in degrees counterclockwise from North
        :return: Array of bearings in degrees with the same size as ixs
        """
        if ref_bearing < 0 or ref_bearing >= 360.0:
            raise ValueError('The reference bearing must be in the range [0, 360)')

        r_ix = np.radians(self.data_set[ix, :])
        r_ixs = np.radians(self.data_set[ixs, :])

        delta_lons = r_ixs[:, 0] - r_ix[0]
        y = np.multiply(np.sin(delta_lons), np.cos(r_ixs[:, 1]))
        x = math.cos(r_ix[1]) * np.sin(r_ixs[:, 1]) - \
            math.sin(r_ix[1]) * np.multiply(np.cos(r_ixs[:, 1]), np.cos(delta_lons))
        bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0 - ref_bearing
        bearings[bearings < 0.0] += 360.0
        return bearings

    def recurse_calculate(self, k):
        """
        Calculates the concave hull using the next value for k while reusing the distances dictionary
        :return: Concave hull
        """
        recurse = ConcaveHull(self.data_set)
        next_k = get_next_k(k)
        # if next_k == -1:
        #     return None
        return recurse.calculate(next_k)

    def calculate(self, k=3):
        """
        Calculates the convex hull of the data set as an array of points
        :param k: Number of nearest neighbors
        :return: Array of points (N, 2) with the concave hull of the data set
        """
        if self.data_set.shape[0] < 3:
            return None

        if self.data_set.shape[0] == 3:
            return self.data_set

        # Make sure that k neighbors can be found
        k = min(k, self.data_set.shape[0])
        # print(f"k = {k}")

        first_point = self.get_lowest_latitude_index(self.data_set)
        current_point = first_point

        # Note that hull and test_hull are matrices (N, 2)
        hull = np.reshape(np.array(self.data_set[first_point, :]), (1, 2))
        test_hull = hull

        # Remove the first point
        self.indices[first_point] = False

        prev_angle = 270    # The initial reference is due west. North is zero, measured clockwise.
        step = 2
        stop = 3  # 2 + k

        while ((current_point != first_point) or (step == 2)) and len(self.indices[self.indices]) > 0:
            if step == stop:
                self.indices[first_point] = True

            knn = self.get_k_nearest(current_point, k)

            # Calculates the bearings between the first_point, and the knn points
            # Returns angles in the same indexing sequence as in knn
            bearings = self.calculate_bearings(current_point, knn, prev_angle)

            # Calculate the candidate indexes (largest angles first)
            candidate_list = np.argsort(-bearings)

            i = 0
            invalid_hull = True

            while invalid_hull and i < len(candidate_list):
                candidate = candidate_list[i]

                # Create a test hull to check if there are any self-intersections
                next_point = np.reshape(self.data_set[knn[candidate]], (1, 2))
                test_hull = np.append(hull, next_point, axis=0)

                line = LineString(test_hull)
                invalid_hull = not line.is_simple
                i += 1

            if invalid_hull:
                return self.recurse_calculate(k)

            prev_angle = self.calculate_bearings(knn[candidate], np.array([current_point]))[0]
            current_point = knn[candidate]
            hull = test_hull

            self.indices[current_point] = False
            step += 1

        poly = Polygon(hull)

        count = 0
        total = self.data_set.shape[0]
        for ix in range(total):
            pt = Point(self.data_set[ix, :])
            if poly.intersects(pt) or pt.within(poly):
                count += 1
            else:
                d = poly.distance(pt)
                if d < 1e-5:
                    count += 1

        if count == total:
            return hull
        else:
            return self.recurse_calculate(k)
