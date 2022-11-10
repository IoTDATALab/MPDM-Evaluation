from itertools import zip_longest

from attr import attrs, attrib, astuple


@attrs
class CloudResource(object):
    bandwidth = attrib(type=float)

    def __add__(self, other):
        if not isinstance(other, CloudResource):
            return NotImplemented
        left = astuple(self)
        right = astuple(other)
        pairs = zip_longest(left, right, fillvalue=0.0)
        values = [a + b for a, b in pairs]
        return CloudResource(*values)

    def __sub__(self, other):
        if not isinstance(other, CloudResource):
            return NotImplemented
        left = astuple(self)
        right = astuple(other)
        pairs = zip(left, right)
        values = [round(a - b,3) for a, b in pairs]
        return CloudResource(*values) if all(map(lambda _: _ >= 0, values))  else None

@attrs(frozen=True)
class CloudModel(object):
    id = attrib(type=int)
    accuracy = attrib(type=float)


@attrs
class Cloud(object):
    resources = attrib(type=CloudResource)
    model = attrib(type=CloudModel)
