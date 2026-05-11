import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    import sys
    import altair as alt

    # utils.py file
    # in: FMA: A Dataset For Music Analysis
    # Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).
    # available under "https://github.com/mdeff/fma"
    from src.FMA.utils import load, get_audio_path
    from src.globals import (
        CSV_FOLDER,
        DATASET_FOLDER,
        TRACKS_PATH,
        AUDIO_FOLDER,
        STEMS_FOLDER,
        UVR_MODEL_PATH,
    )
    return DATASET_FOLDER, alt, os, pd


@app.cell
def _(DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey")
    return (SURVEY_FOLDER,)


@app.cell
def _(pd):
    def parse_js_date(series):
        # Remove " (Coordinated Universal Time)" and similar parenthetical suffixes
        cleaned = series.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
        return pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z")
    return (parse_js_date,)


@app.cell
def _(SURVEY_FOLDER, os, parse_js_date, pd):
    participants = pd.read_csv(os.path.join(SURVEY_FOLDER, "participants.csv"))
    surveyAnswers = pd.read_csv(os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"))
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    participants[participants.surveyCompleted]
    return participants, surveyAnswers


@app.cell
def _(participants):
    participants["completionTime"] = (
        participants["editDate"] - participants["createDate"]
    )
    participants["completionMinutes"] = (
        participants["completionTime"].dt.total_seconds() / 60
    )
    participants[participants.surveyCompleted]
    return


@app.cell
def _(participants):
    participants[participants.surveyCompleted].completionMinutes.to_list()
    return


@app.cell
def _(alt, participants):
    _chart = (
        alt.Chart(
            participants[participants.surveyCompleted][["completionMinutes"]]
        )  # <-- replace with data
        .mark_bar()
        .encode(
            x=alt.X(
                "completionMinutes",
                type="quantitative",
                bin=True,
                title="completionMinutes",
            ),
            y=alt.Y("count()", type="quantitative", title="Number of records"),
            tooltip=[
                alt.Tooltip(
                    "completionMinutes",
                    type="quantitative",
                    bin=True,
                    title="completionMinutes",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "count()",
                    type="quantitative",
                    format=",.0f",
                    title="Number of records",
                ),
            ],
        )
        .properties(width="container")
        .configure_view(stroke=None)
    )
    _chart
    return


@app.cell
def _(surveyAnswers):
    surveyAnswers
    return


if __name__ == "__main__":
    app.run()
