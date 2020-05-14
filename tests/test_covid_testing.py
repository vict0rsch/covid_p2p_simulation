import datetime
import hashlib
import os
import pickle
import unittest
import zipfile
from tempfile import TemporaryDirectory
from pathlib import Path

from covid19sim.run import simulate
from covid19sim.base import Event
from covid19sim.utils import extract_tracker_data
from covid19sim.configs.exp_config import ExpConfig


if __name__ == "__main__":
    path = Path(__file__).parent
    ExpConfig.load_config(path / "test_configs" / "naive_local.yml")
    outfile = None
    n_people = 100
    simulation_days = 10
    init_percent_sick = 0.02
    monitors, tracker = simulate(
        n_people=n_people,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=simulation_days,
        outfile=outfile,
        init_percent_sick=init_percent_sick,
        out_chunk_size=500,
    )
    monitors[0].dump()
    monitors[0].join_iothread()

    data = extract_tracker_data(tracker, ExpConfig)
