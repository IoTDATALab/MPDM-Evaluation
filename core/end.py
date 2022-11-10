from typing import List

from attr import attrs, attrib, asdict
from switchlang import switch

from .cloud import CloudModel
from .edge import EdgeModel, EdgeNode


@attrs(frozen=True)
class StreamDemands(object):
    bandwidth = attrib(type=float, default=0)


@attrs
class StreamDecision(object):
    id = attrib(type=int, default=0)
    type = attrib(type=str, default=None)
    edge_model = attrib(type=EdgeModel, default=None)
    edge_node = attrib(type=EdgeNode, default=None)
    cloud_model = attrib(type=CloudModel, default=None)
    alpha = attrib(type=float, default=0)
    accuracy = attrib(type=float, default=0)

    def __attrs_post_init__(self):
        with switch(self.type) as s:
            s.case('PureEdge', self.__pure_edge_processing)
            s.case('PureCloud', self.__pure_cloud_processing)
            # Edge-Cloud Synergy
            s.case('ECS-CLB', self.__ecs_processing)
            s.default(lambda: None)

    @alpha.validator
    def check(self, _, value):
        if value < 0 or value > 1:
            raise ValueError("The alpha should be in range [0,1].")

    def __pure_edge_processing(self):
        self.accuracy = self.edge_model.accuracy

    def __pure_cloud_processing(self):
        self.accuracy = self.cloud_model.accuracy

    def __ecs_processing(self):
        self.accuracy = round(self.alpha * self.cloud_model.accuracy + (1 - self.alpha) * self.edge_model.accuracy, 4)

    def demands(self, stream):
        if self.type == "PureCloud":
            demands = {"cpu": 0, "ram": 0, "gpu": 0, "edge_bandwidth": 0,
                       'cloud_bandwidth': stream.demands.bandwidth}
        else:
            demands = asdict(self.edge_model.demands)
            if self.type == "PureEdge":
                demands['edge_bandwidth'] = stream.demands.bandwidth
                demands['cloud_bandwidth'] = 0
            elif self.type == "ECS-CLB":
                demands['edge_bandwidth'] = stream.demands.bandwidth * ( 1 - self.alpha)
                demands['cloud_bandwidth'] = stream.demands.bandwidth - demands['edge_bandwidth']
        return list(demands.values())


@attrs
class StreamState(object):
    accuracy = attrib(type=float, default=0)
    served = attrib(type=bool, default=False)
    bandwidth = attrib(type=float,default=0)

    def update(self, decision, bandwidth=0):
        if decision is None:
            self.served = False
            self.accuracy = 0
        else:
            self.served = True
            self.accuracy = decision.accuracy
        self.bandwidth = bandwidth

@attrs
class Stream(object):
    id = attrib(type=int)
    demands = attrib(type=StreamDemands)
    decision = attrib(type=StreamDecision, default=None)
    state = attrib(type=StreamState, default=StreamState())


@attrs
class End(object):
    streams = attrib(factory=list, type=List[Stream])
