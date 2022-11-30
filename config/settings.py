from pathlib import Path
from dataclasses import dataclass
from astropy import units
from astropy.time import Time
import argparse

__all__ = [
    "PROJECT_ROOT", 
    "DATA_PATH", 
    "CACHE_PATH",
    "RANDOM_STATE", 
    "START_TIME", 
    "END_TIME",
    "NUM_TIMES",
    "DO_ANALYSIS"
    ]

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_PATH = PROJECT_ROOT / 'data'

CACHE_PATH = PROJECT_ROOT / 'cache'

RANDOM_STATE: int = 42

#START_TIME = Time("2003-06-01T12:00:00.000", format="fits", scale="utc")
START_TIME = Time("2000-01-01T12:00:00.000", format="fits", scale="utc")
END_TIME = START_TIME + 1 * units.min

CROSS_START_TIME = START_TIME + 1 * units.min + 10 * units.s
CROSS_END_TIME = START_TIME + 1 * units.min + 15 * units.s

NUM_TIMES: int = 1000

DO_ANALYSIS = True