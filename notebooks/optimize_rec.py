import numpy as np
import dill
from collections import defaultdict
import datetime
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pdb


def data_str(v):
    """
    Get a string representation of a data value:
      v itself if not list or dict or tuple
      len(v) otherwise

    Args:
        v (Any): value to print

    Returns:
        str: string for v
    """
    return v if not isinstance(v, (list, dict, tuple)) else "{} items".format(len(v))


def get_category(risk_attr):
    """
    Assigns a category to a given risk attribute,
    that is to a human at a given hour of a given day

    Args:
        risk_attr (dict): dictionnary representing a human's risk at a given time

    Returns:
        str: category for this person
    """
    if risk_attr["exposed"]:
        return "A"

    if risk_attr["infectious"] and risk_attr["symptoms"] == 0:
        return "B"

    if risk_attr["infectious"] and risk_attr["symptoms"] > 0:
        return "C"

    if risk_attr["test"]:
        return "D"

    if risk_attr["order_1_is_exposed"]:
        return "E"

    if risk_attr["order_1_is_presymptomatic"]:
        return "H"

    if risk_attr["order_1_is_symptomatic"]:
        return "I"

    if risk_attr["order_1_is_tested"]:
        return "J"

    return "K"


def sample_thresholds(n=1):
    """
    Samples uniformly a set of thresholds:
    r1 = uniform(0, 1)
    r2 = uniform(r1, 1)
    r3 = uniform(r2, 1)

    test:
    for r1, r2, r3 in sample_thresholds(1000):
        assert 0 < r1 < r2 < r3 < 1

    Args:
        n (int, optional): Number of thresholds triplets to sample. Defaults to 1.

    Returns:
        np.array: nx3 array of sampled thresholds
    """
    r1 = np.random.uniform(0, 0.4, size=n)
    r2 = np.random.uniform(r1, 1, size=n)
    r3 = np.random.uniform(r2, 1, size=n)
    return np.array([r1, r2, r3]).T


def predict_color_id(risks, thresholds):
    """
    Get the color_id prediction for the risks according to thresholds:

        risk < thresholds[0] => 0
        risk > thresholds[1] => 1
        risk > thresholds[2] => 2
        risk > thresholds[3] => 3

    Args:
        risks (list): float, risks
        thresholds (list): 3 values describing the risk-thresholds for colors

    Returns:
        np.array: array as long as risks where they have been categorized as ints
    """
    predictions = np.zeros_like(risks)
    for i, r in enumerate(thresholds):
        predictions[risks > r] = i + 1
    return predictions.astype(int)


if __name__ == "__main__":

    # -----------------------
    # -----  Constants  -----
    # -----------------------
    ptrace = pdb.set_trace  # debug only
    filename = "/Users/victor/Downloads/tracker_data_n_200_seed_0_20200509-090220_.pkl"
    category_to_color = {
        "A": "RED",
        "B": "RED",
        "C": "RED",
        "D": "RED",
        "E": "YELLOW",
        "F": "RED",
        "G": "RED",
        "H": "ORANGE",
        "I": "ORANGE",
        "J": "RED",
        "K": "GREEN",
    }
    color_to_id = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    id_to_color = {v: k for k, v in color_to_id.items()}

    # -----------------------------
    # -----  Parse Arguments  -----
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=filename)
    parser.add_argument("--samples", type=int, default=100)
    opts = parser.parse_args()

    assert Path(opts.data).exists()
    assert opts.samples > 0

    # ---------------------------------------
    # -----  Prepare Ground Truth Data  -----
    # ---------------------------------------
    data = dill.load(open(opts.data, "rb"))
    print("Loaded", opts.data)
    print(
        "\n".join("{:35}: {}".format(k, data_str(v)) for k, v in sorted(data.items()))
    )

    risk_attributes = data["risk_attributes"]
    risks = [r["risk"] for r in risk_attributes]
    categories = [get_category(r) for r in risk_attributes]
    colors = [category_to_color[c] for c in categories]
    color_ids = [color_to_id[c] for c in colors]

    models = sample_thresholds(opts.samples)

    max_acc = 0
    max_thresholds = None

    for i, model in enumerate(models):
        predictions = predict_color_id(risks, model)
        cm = confusion_matrix(color_ids, predictions)
        acc = (color_ids == predictions).mean()
        print("Test model", i, end='\r')
        if max_acc < acc:
            max_acc = acc
            max_thresholds = model
            print(">>> ------------------ <<<")
            print(cm)
            print("{} % acc".format(acc))
            print("=> NEW max acc, for threshold {}".format(model))
