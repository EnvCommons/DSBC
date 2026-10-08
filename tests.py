"""Tests for the reference corrections applied in DSBC.list_tasks.

    pytest tests.py

The derivation tests recompute each corrected reference from the source CSVs
(large-traversaal/DSBC-DataFiles on Hugging Face) and are skipped unless those
files are in datasets/.
"""
from pathlib import Path

import pandas as pd
import pytest

import dsbc
from dsbc import DSBC

DATASETS = Path(__file__).parent / "datasets"

ORIGINAL_ROWS = {
    49: (
        "COVID Dataset",
        "Using valid death dates only, aggregate total deaths by ISO week-of-year across the "
        "entire dataset (combining all years), plot deaths versus ISO week number, and then "
        "determine the longest consecutive run of ISO weeks that achieve the global maximum "
        "weekly death count. Report the start and end ISO week numbers in the format \"X-Y weeks\".",
        "15-16 weeks",
    ),
    191: (
        "PRODUCTION Dataset",
        "Using the most recent ten calendar years present in the data (inclusive), compute for "
        "each region the total seeded area measured in hectares over that period, then identify "
        "the region with the smallest such total. Answer with the region abbreviation, then a "
        "comma, then the total area as an integer (e.g., AB,123456).",
        "NS, 940250",
    ),
    219: (
        "SALES Dataset",
        "Identify the five stores whose weekly sales have the strongest relationship with the "
        "calendar day of the month by computing, for each store, the absolute Pearson "
        "correlation between its weekly sales and the day number within the month extracted "
        "from the date. Return the five store numbers with the largest absolute correlations, "
        "listed from highest to lowest, separated by commas with no spaces.",
        "21, 7, 39, 32, 16",
    ),
}


ORIGINAL_ROWS_2 = {
    24: (
        "AQI Dataset",
        "Considering all records in the dataset, determine which calendar month, derived from "
        "each record\u2019s timestamp, has the lowest average air temperature. If multiple months "
        "tie, choose the earliest month in the year. Answer with the English month name.",
        "March",
    ),
    33: (
        "AQI Dataset",
        "What were the highest and lowest temperatures recorded over time? Answer with two "
        "floats separated by a comma, highest first.",
        "59.9, 22.43",
    ),
    96: (
        "INSURANCE Dataset",
        "What is the shape of the BMI distribution among customers in this dataset? Answer "
        "with one of: normal,left-skewed,right-skewed,uniform",
        "There is a nearly normal distribution of BMI among customers",
    ),
    221: (
        "SALES Dataset",
        "Which holiday week (calendar week number) recorded the lowest total weekly sales, and "
        "which holiday is it? Answer in the format \u201cweekNN,holidayname\u201d (two-digit week "
        "number, lowercase holiday name with no spaces). For example: week13,easter",
        "Week 52, new year",
    ),
}


def write_dataset(tmp_path, rows):
    records = []
    for i in range(max(rows) + 2):
        dataset, question, answer = rows.get(i, ("STOCKS Dataset", f"Question {i}?", f"{i}"))
        records.append({"Dataset": dataset, "Question_Rewritten": question, "Response_Expected": answer})
    path = tmp_path / "dataset.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_corrections_replace_the_references(tmp_path, monkeypatch):
    monkeypatch.setattr(dsbc, "DATASET_PATH", write_dataset(tmp_path, ORIGINAL_ROWS))
    tasks = DSBC.list_tasks("train")
    assert len(tasks) == 221
    assert [t["task_id"] for t in tasks] == list(range(221))
    assert tasks[49]["answer"] == "25-25 weeks"
    assert tasks[191]["answer"] == "NS, 846100"
    assert tasks[219]["answer"] == "42, 28, 38, 43, 33"
    assert tasks[49]["question"] == ORIGINAL_ROWS[49][1]
    assert tasks[191]["question"] == ORIGINAL_ROWS[191][1]
    assert "summed over all departments" in tasks[219]["question"]
    assert tasks[50] == {"task_id": 50, "dataset": "STOCKS Dataset", "question": "Question 50?", "answer": "50"}


def test_corrections_of_tasks_24_33_96_221(tmp_path, monkeypatch):
    monkeypatch.setattr(dsbc, "DATASET_PATH", write_dataset(tmp_path, ORIGINAL_ROWS_2))
    tasks = DSBC.list_tasks("train")
    assert [t["task_id"] for t in tasks] == list(range(223))
    assert tasks[24]["answer"] == "January"
    assert tasks[33]["answer"] == "31.7, 16.45"
    assert tasks[96]["answer"] == "normal"
    assert tasks[221]["answer"] == "week52, christmas"
    assert "`AT (degree C)`" in tasks[24]["question"]
    assert "`AT (degree C)`" in tasks[33]["question"]
    assert "absolute sample skewness" in tasks[96]["question"]
    assert tasks[221]["question"] == ORIGINAL_ROWS_2[221][1]
    assert tasks[25] == {"task_id": 25, "dataset": "STOCKS Dataset", "question": "Question 25?", "answer": "25"}


def test_correction_skipped_when_the_row_does_not_match(tmp_path, monkeypatch):
    rows = {49: ("COVID Dataset", "Some other question?", "seven")}
    monkeypatch.setattr(dsbc, "DATASET_PATH", write_dataset(tmp_path, rows))
    task = DSBC.list_tasks("train")[49]
    assert (task["question"], task["answer"]) == ("Some other question?", "seven")


def test_corrected_references_take_the_llm_grader_path():
    # A comma-joined list without spaces parses as one large number, which would
    # be compared numerically against the last number in the model's answer.
    for _, answer, _ in dsbc.CORRECTIONS.values():
        assert dsbc._to_float(answer) is None


def source(name):
    path = DATASETS / f"{name}_TRAIN.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    return pd.read_csv(path)


def test_derive_49():
    df = source("COVID")
    died = pd.to_datetime(df.loc[df["DATE_DIED"] != "9999-99-99", "DATE_DIED"], format="%d/%m/%Y")
    weekly = died.dt.isocalendar().week.value_counts()
    peak_weeks = sorted(weekly[weekly == weekly.max()].index)
    assert peak_weeks == [25]
    assert dsbc.CORRECTIONS[49][1] == "25-25 weeks"


def test_derive_191():
    df = source("PRODUCTION")
    last = df["REF_DATE"].max()
    years = list(range(last - 9, last + 1))
    assert set(years) <= set(df["REF_DATE"])
    totals = df[df["REF_DATE"].isin(years)].groupby("GEO")["Seeded area (hectares)"].sum()
    assert dsbc.CORRECTIONS[191][1] == f"{totals.idxmin()}, {int(totals.min())}"


def test_derive_219():
    df = source("SALES")
    df["Date"] = pd.to_datetime(df["Date"])
    weekly = df.groupby(["Store", "Date"], as_index=False)["Weekly_Sales"].sum()
    weekly["Day"] = weekly["Date"].dt.day
    corr = weekly.groupby("Store")[["Weekly_Sales", "Day"]].apply(
        lambda g: abs(g["Weekly_Sales"].corr(g["Day"]))
    )
    top = corr.sort_values(ascending=False).index[:5]
    assert dsbc.CORRECTIONS[219][1] == ", ".join(str(s) for s in top)


def test_derive_24_and_33():
    df = source("AQI")
    month = pd.to_datetime(df["From Date"]).dt.month
    monthly = df.groupby(month)["AT (degree C)"].mean()
    assert dsbc.CORRECTIONS[24][1] == pd.Timestamp(2000, monthly.idxmin(), 1).month_name()
    at = df["AT (degree C)"]
    assert dsbc.CORRECTIONS[33][1] == f"{at.max()}, {at.min()}"


def test_derive_96():
    skew = source("INSURANCE")["bmi"].skew()
    assert abs(skew) < 0.5
    assert dsbc.CORRECTIONS[96][1] == "normal"


def test_derive_221():
    # Holiday weeks named in the source competition's data description.
    christmas = pd.to_datetime(["2010-12-31", "2011-12-30", "2012-12-28"])
    df = source("SALES")
    df["Date"] = pd.to_datetime(df["Date"])
    totals = df[df["IsHoliday"]].groupby("Date")["Weekly_Sales"].sum()
    lowest = totals.idxmin()
    assert lowest in christmas
    assert dsbc.CORRECTIONS[221][1] == f"week{lowest.isocalendar().week:02d}, christmas"
