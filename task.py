from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

class Label(ABC):

    @classmethod
    @abstractmethod
    def collate(cls, labels):
        raise NotImplementedError

@dataclass
class Task(ABC):
    data : Any
    """The input data, for instance an image represented as a numpy array.
    """

    @abstractmethod
    def show(self):
        pass


class SegmentationLabel(Label): pass
@dataclass
class SegmentationTask(Task):
    label : SegmentationLabel


class ObjectDetectionLabel(Label): pass
@dataclass
class ObjectDetectionTask(Task):
    label : ObjectDetectionLabel


class PersonPoseLabel(Label): pass
@dataclass
class PersonPoseTask(Task):
    label : PersonPoseTask


@dataclass
class MultiTask(Task):
    labels : List[Label]
