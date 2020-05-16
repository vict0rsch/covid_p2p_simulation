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
    ml = max([len(k) for k in dic.keys()] + [0]) + 2
    aligned = "{:" + str(ml) + "}"
    print(
        "{}:\n   ".format(title),
        "\n    ".join((aligned + ": {}").format(k, v) for k, v in items)
        if items
        else "<No data>",
    )


def date_str(date):
    return str(date).split()[0]


def compute_positivity_rate(positive_tests_per_day, negative_tests_per_day, average=3):
    rates = []
    days = []
    q = deque(maxlen=average)
    d = deque(maxlen=average)
    for i, day in enumerate(positive_tests_per_day):
        pt = positive_tests_per_day[day]
        nt = negative_tests_per_day[day]
        if pt + nt:
            q.append(pt / (nt + pt))
        else:
            q.append(0)
        d.append(day)
        rates.append(np.mean(q))
        days.append(" - ".join([_.replace("2020-", "").replace("-", "/") for _ in d]))
    return {d: "{:.3f}%".format(r * 100) for d, r in zip(days, rates)}


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # https://coronavirus.jhu.edu/testing/testing-positivity
    # https://www.canada.ca/content/dam/phac-aspc/documents/services/diseases/2019-novel-coronavirus-infection/surv-covid19-epi-update-eng.pdf
    path = Path(__file__).parent
    # test_covid_test = no intervention
    ExpConfig.load_config(path / "test_configs" / "test_covid_test.yml")
    outfile = None

    # ----------------------------
    # -----  Run Simulation  -----
    # ----------------------------
    n_people = 1000
    simulation_days = 60
    init_percent_sick = 0.002
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    monitors, tracker, city = simulate(
        n_people=n_people,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
        init_percent_sick=init_percent_sick,
        out_chunk_size=500,
    )

    # ---------------------------
    # -----  Retreive Data  -----
    # ---------------------------
    data = extract_tracker_data(tracker, ExpConfig)
    tm = data["test_monitor"]
    days = [
        date_str(start_time + datetime.timedelta(days=i))
        for i in range(simulation_days)
    ]

    # -------------------------
    # -----  Daily Stats  -----
    # -------------------------
    cum_removed = data["r"]
    i_for_day = {days[i]: d for i, d in enumerate(data["i"])}

    # ------------------------------
    # -----  Positivity Rates  -----
    # ------------------------------
    positive_tests_per_day = {d: 0 for d in days}
    negative_tests_per_day = {d: 0 for d in days}
    for m in tm:
        if m["result"] == "positive":
            positive_tests_per_day[date_str(m["timestamp"])] += 1
        if m["result"] == "negative":
            negative_tests_per_day[date_str(m["timestamp"])] += 1
    tests_per_human = Counter([m["name"] for m in tm])
    tests_per_day = Counter([date_str(m["timestamp"]) for m in tm])
    positivity_rates = compute_positivity_rate(
        positive_tests_per_day, negative_tests_per_day, average=3
    )
    positives = len([m for m in tm if m["result"] == "positive"])
    negatives = len([m for m in tm if m["result"] == "negative"])
    if negatives + positives:
        positive_rate = positives / (negatives + positives)
    else:
        positive_rate = "Warning: no test done"

    # ----------------------------
    # -----  Multiple Tests  -----
    # ----------------------------
    people_with_several_tests = {k: v for k, v in tests_per_human.items() if v > 1}
    multi_test_dates = {}
    for m in tm:
        if m["name"] in people_with_several_tests:
            old = multi_test_dates.get(m["name"], [])
            new = "{} : {}".format(date_str(m["timestamp"]), m["result"])
            multi_test_dates[m["name"]] = old + [new]

    # ----------------------
    # -----  Symptoms  -----
    # ----------------------
    symptoms = Counter([s for m in tm for s in m["symptoms"]])
    symptoms_positive = Counter(
        [s for m in tm for s in m["symptoms"] if m["result"] == "positive"]
    )
    symptoms_negative = Counter(
        [s for m in tm for s in m["symptoms"] if m["result"] == "negative"]
    )

    # --------------------
    # -----  Prints  -----
    # --------------------
    print("\n" + "-" * 50 + "\n" + "-" * 50)
    print("Cumulative removed per day: ", cum_removed)
    print("Test events: ", len(tm))
    print("Individuals: ", len(set(m["name"] for m in tm)))
    print_dict("Tests per day", tests_per_day)
    if any(v > city.max_capacity_per_test_type["lab"] for v in tests_per_day.values()):
        print(">>> ----- WARNING ----- <<<")
        print(
            " There are days with abnormal number of tests (max cap = {})\n".format(
                city.max_capacity_per_test_type["lab"]
            )
        )
    print(
        "Results last day ( N | P | P / (N+P) ): ", negatives, positives, positive_rate
    )
    print("For reference, May 14th in Canada: ", 1104855, 66536, 66536 / 1172796)
    print_dict("Positivity rates", positivity_rates)
    print_dict("All symptoms", symptoms, is_sorted="desc")
    print_dict("Symptoms when positive tests", symptoms_positive, is_sorted="desc")
    print_dict("Symptoms when negative tests", symptoms_negative, is_sorted="desc")
