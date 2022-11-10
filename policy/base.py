from copy import deepcopy

from attr import attrs, attrib
from sortedcontainers import SortedList

from core import System, StreamDecision
import numpy as np


@attrs
class BasePolicy(object):
    system = attrib(type=System)

    def __attrs_post_init__(self):
        self.system = deepcopy(self.system)

        cloud_model = self.system.cloud.model
        alpha = self.system.alpha
        stream_decisions = SortedList(key=lambda _: -_.accuracy)

        i = 0

        for edge_model in self.system.edge.models:
            if "PureEdge" in self.system.paradigms:
                stream_decision = StreamDecision(id=i,
                                                 type="PureEdge",
                                                 edge_model=edge_model)
                stream_decisions.add(stream_decision)
                i += 1
            if "ECS-CLB" in self.system.paradigms:
                stream_decision = StreamDecision(id=i,
                                                 type="ECS-CLB",
                                                 edge_model=edge_model,
                                                 cloud_model=cloud_model,
                                                 alpha=alpha)
                stream_decisions.add(stream_decision)
                i += 1
        if "PureCloud" in self.system.paradigms:
            stream_decision = StreamDecision(id=i,
                                             type="PureCloud",
                                             cloud_model=cloud_model)
            stream_decisions.add(stream_decision)
        self.stream_decisions = list(stream_decisions)



    def normalize(self, value, min_value, max_value):
        ratio=0.99999999
        result = (value - min_value*ratio) / (max_value - min_value*ratio+np.finfo(np.float).eps)
        return result
