import numpy as np
from geo.math import vec_haversine
from typing import List
from collections import Counter


class OSMNode(object):

    def __init__(self, nid, lat, lon):
        self.nid = nid
        self.lat = lat
        self.lon = lon
        self.name = ""

    def __repr__(self):
        return "OSMNode(nid={0}, lat={1}, lon={2}, name={3})"\
            .format(self.nid,
                    self.lat,
                    self.lon,
                    self.name)


class OSMWay(object):

    def __init__(self, wid, nodes, tags):
        self.wid = wid
        self.nodes = set(nodes)
        self.tags = tags

    def has_node(self, nid):
        return nid in self.nodes

    def is_highway(self):
        return "highway" in self.tags

    def has_name(self):
        return "name" in self.tags

    def __repr__(self):
        if self.has_name() and self.is_highway():
            return "OSMWay(wid={0}, name={1})"\
                .format(self.wid, self.tags['name'])
        else:
            return "OSMWay(wid={0})".format(self.wid)


class OSMNet(object):

    def __init__(self, nodes: List[OSMNode], ways: List[OSMWay]):
        self.ref_nodes = []
        self.nodes = nodes
        self.ways = ways
        self.node_idx = {node.nid: node for node in nodes}
        self.way_nodes = {way.wid: [self.node_idx[nid] for nid in way.nodes]
                          for way in ways}
        self.named_highways = [way for way in self.ways
                               if way.is_highway() and way.has_name()]
        for way in self.named_highways:
            for node in [self.node_idx[n] for n in way.nodes]:
                node.name = way.tags['name']
                self.ref_nodes.append(node)

    def get_knn(self, lat: float, lon: float, k: int = 5) -> List[OSMNode]:
        lats = np.array([n.lat for n in self.ref_nodes])
        lons = np.array([n.lon for n in self.ref_nodes])
        dist = vec_haversine(lats, lons, lat, lon)
        return np.argsort(dist)[:k]

    def get_way_name(self, lat: float, lon: float, k: int = 7) -> str:
        lats = np.array([n.lat for n in self.ref_nodes])
        lons = np.array([n.lon for n in self.ref_nodes])
        dist = vec_haversine(lats, lons, lat, lon)
        idx = np.argsort(dist)[:k]
        names = [self.ref_nodes[i].name for i in idx]
        cnt = Counter(names)
        cmn = cnt.most_common(2)
        return " / ".join([n[0] for n in cmn])

    def get_name(self, points: np.ndarray, k: int = 5) -> str:
        knn = []
        for pt in points:
            knn.extend(self.get_knn(pt[0], pt[1], k).tolist())

        names = [self.ref_nodes[n].name for n in knn]

        cnt = Counter(names)
        cmn = cnt.most_common(2)
        return " / ".join([n[0] for n in cmn])

    @classmethod
    def from_overpass(cls, data):
        nodes = [OSMNode(n['id'], n['lat'], n['lon']) for n in data['elements']
                 if n['type'] == 'node']
        ways = [OSMWay(w['id'], w['nodes'], w['tags']) for w in data['elements']
                if w['type'] == 'way']
        return cls(nodes, ways)
