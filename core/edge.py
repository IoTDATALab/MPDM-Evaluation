from typing import List
from itertools import zip_longest

from attr import attrs, attrib, astuple


@attrs
class EdgeResource(object):
    cpu = attrib(type=float, default=0)
    ram = attrib(type=float, default=0)
    gpu = attrib(type=float, default=0)
    bandwidth = attrib(type=float, default=0)

    def __add__(self, other):
        if not isinstance(other, EdgeResource):
            return NotImplemented
        left = astuple(self)
        right = astuple(other)
        pairs = zip_longest(left, right, fillvalue=0.0)
        values = [a + b for a, b in pairs]
        return EdgeResource(*values)

    def __sub__(self, other):
        if not isinstance(other, EdgeResource):
            return NotImplemented
        left = astuple(self)
        right = astuple(other)
        pairs = zip_longest(left, right, fillvalue=0.0)
        values = [round(a - b, 3) for a, b in pairs]
        return EdgeResource(*values) if all(map(lambda _: _ >= 0, values)) else None


@attrs
class EdgeNode(object):
    id = attrib(type=int)
    resources = attrib(type=EdgeResource)


@attrs(frozen=True)
class EdgeModelDemands:
    cpu = attrib(type=float, default=0)
    ram = attrib(type=float, default=0)
    gpu = attrib(type=float, default=0)


@attrs(frozen=True)
class EdgeModel(object):
    id = attrib(type=int)
    demands = attrib(type=EdgeModelDemands)
    accuracy = attrib(type=float)


@attrs
class Edge(object):
    nodes = attrib(factory=list, type=List[EdgeNode])
    models = attrib(factory=list, type=List[EdgeModel])
