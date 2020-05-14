import datetime
import hashlib
import os
import pickle
import unittest
import zipfile
from collections import Counter, deque
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from covid19sim.base import Event
from covid19sim.configs.exp_config import ExpConfig
from covid19sim.run import simulate
from covid19sim.utils import extract_tracker_data


def print_dict(title, dic, is_sorted=None):
    if not is_sorted:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[1])
        if is_sorted == "desc":
            items = reversed(items)
    ml = max(len(k) for k in dic.keys()) + 2
    aligned = "{:" + str(ml) + "}"
    print(
        "{}:\n   ".format(title),
        "\n    ".join((aligned + ": {}").format(k, v) for k, v in items),
    )


def compute_positivity_rate(tests_per_day, n_people, average=3):
    rates = []
    days = []
    q = deque(maxlen=average)
    d = deque(maxlen=average)
    for i, (day, count) in enumerate(tests_per_day.items()):
        q.append(count)
        d.append(day)
        rates.append(np.mean(q) / n_people)
        days.append(" - ".join([_.replace("2020-", "").replace("-", "/") for _ in d]))
    return {d: "{:.3f}%".format(r * 100) for d, r in zip(days, rates)}


if __name__ == "__main__":
    # https://coronavirus.jhu.edu/testing/testing-positivity
    path = Path(__file__).parent
    ExpConfig.load_config(path / "test_configs" / "naive_local.yml")
    outfile = None
    n_people = 1000
    simulation_days = 30
    init_percent_sick = 0.01
    monitors, tracker = simulate(
        n_people=n_people,
        start_time=datetime.datetime(2020, 2, 28, 0, 0),
        simulation_days=simulation_days,
        outfile=outfile,
        init_percent_sick=init_percent_sick,
        out_chunk_size=500,
    )

    data = extract_tracker_data(tracker, ExpConfig)

    cum_R = np.cumsum(data["r"])

    tm = data["test_monitor"]
    mm = tm[0]
    h = mm["human"]

    tests_per_human = Counter([m["human"].name for m in tm])
    tests_per_day = Counter([str(m["timestamp"]).split()[0] for m in tm])
    positivity_rates = compute_positivity_rate(tests_per_day, n_people, average=3)
    people_with_several_tests = {k: v for k, v in tests_per_human.items() if v > 1}

    multi_test_dates = {}
    for m in tm:
        if m["human"].name in people_with_several_tests:
            old = multi_test_dates.get(m["human"].name, [])
            new = "{} : {}".format(str(m["timestamp"]).split()[0], m["result"])
            multi_test_dates[m["human"].name] = old + [new]

    positives = len([m for m in tm if m["result"] == "positive"])
    negatives = len([m for m in tm if m["result"] == "negative"])
    negatives_rate = negatives / (negatives + positives)
    symptoms = Counter([s for m in tm for s in m["human"].symptoms])
    symptoms_positive = Counter(
        [s for m in tm for s in m["human"].symptoms if m["result"] == "positive"]
    )
    symptoms_negative = Counter(
        [s for m in tm for s in m["human"].symptoms if m["result"] == "negative"]
    )

    print("Cumulative removed per day: ", cum_R)
    print("Test events: ", len(tm))
    print("Individuals: ", len(set(m["human"] for m in tm)))
    print_dict("Multi-tests humans", multi_test_dates)
    print_dict("Tests per day", tests_per_day)
    print("Results ( N | P | N / (N+P) ): ", negatives, positives, negatives_rate)
    print_dict("All symptoms", symptoms, is_sorted="desc")
    print_dict("Symptoms when positive tests", symptoms_positive, is_sorted="desc")
    print_dict("Symptoms when negative tests", symptoms_negative, is_sorted="desc")
    print_dict("Positivity rates", positivity_rates)
