from enum import Enum


class Status(str, Enum):
    NotStarted = "NotStarted"
    Running = "Running"
    Succeeded = "Succeeded"
    Failed = "Failed"


class Voice(str, Enum):
    Male1 = "Male1"
    Female1 = "Female1"
