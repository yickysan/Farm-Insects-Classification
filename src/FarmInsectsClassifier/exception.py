import sys
from FarmInsectsClassifier.logger import logging


class DataIngestionError(Exception):
    pass

class BaseModelPrepError(Exception):
    pass