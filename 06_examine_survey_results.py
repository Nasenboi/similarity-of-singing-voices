import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    #  Import Python Packages
    """)
    return


@app.cell
def _():
    import os
    import sys

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    # utils.py file
    # in: FMA: A Dataset For Music Analysis
    # Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset for Music Analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR).
    # available under "https://github.com/mdeff/fma"
    from src.FMA.utils import get_audio_path, load
    from src.globals import AUDIO_FOLDER, CSV_FOLDER, DATASET_FOLDER, STEMS_FOLDER, TRACKS_PATH, UVR_MODEL_PATH

    return DATASET_FOLDER, alt, mo, os, pd


@app.cell
def _(DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey")
    return (SURVEY_FOLDER,)


@app.cell
def _(pd):
    def parse_js_date(series):
        cleaned = series.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
        return pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z")

    return (parse_js_date,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load all Datasets
    """)
    return


@app.cell
def _(SURVEY_FOLDER, os, parse_js_date, pd):
    participants = pd.read_csv(os.path.join(SURVEY_FOLDER, "participants.csv"), index_col="_id")
    surveyQuestions = pd.read_csv(os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"), index_col="_id")
    surveyAnswers = pd.read_csv(os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"), index_col="_id")
    songs = pd.read_csv(os.path.join(SURVEY_FOLDER, "songs.csv"), index_col="_id")
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    participants[participants.surveyCompleted]
    return participants, surveyAnswers, surveyQuestions


@app.cell
def _(participants):
    participants["completionTime"] = participants["editDate"] - participants["createDate"]
    participants["completionMinutes"] = participants["completionTime"].dt.total_seconds() / 60
    participants[participants.surveyCompleted]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Initial Statistics
    """)
    return


@app.cell
def _(participants, surveyAnswers, surveyQuestions):
    ab_ratio_a = 100 * len(surveyAnswers[surveyAnswers.answer_1 == "A"]) / len(surveyAnswers)
    ab_ratio_b = 100 * len(surveyAnswers[surveyAnswers.answer_1 == "B"]) / len(surveyAnswers)

    instruments_on_yes = 100 * len(surveyAnswers[surveyAnswers.backgroundMusic]) / len(surveyAnswers)
    instruments_on_no = 100 * len(surveyAnswers[~surveyAnswers.backgroundMusic]) / len(surveyAnswers)

    print(f"""
    Total number of participants: {len(participants)}
    Number of completed surveys:  {len(participants[participants.surveyCompleted])}
    Number of incomple surveys:   {
        len(participants[~participants.surveyCompleted])
    }

    Median completion time: {
        round(participants.completionMinutes.median())
    } minutes

    Total number of questions: {len(surveyQuestions)} 
    Number of questions with no answers: {
        surveyQuestions[
            ~surveyQuestions.index.isin(surveyAnswers["questionID"])
        ].shape[0]
    }
    Number of questions with multiple answers: {
        surveyAnswers.groupby("questionID").size().loc[lambda x: x > 1].shape[0]
    }

    Total number of answers: {len(surveyAnswers)}
    Answer A/B: {ab_ratio_a:.1f}% A; {ab_ratio_b:.1f}% B
    Instruments on:   {instruments_on_yes:.1f}% Yes; {instruments_on_no:.1f}% No
    """)
    return


@app.cell
def _(alt, participants):
    _chart = (
        alt.Chart(participants[participants.surveyCompleted][["completionMinutes"]])  # <-- replace with data
        .mark_bar()
        .encode(
            x=alt.X(
                "completionMinutes",
                type="quantitative",
                bin=alt.Bin(step=5),
                title="completionMinutes",
            ),
            y=alt.Y("count()", type="quantitative", title="Number of records"),
            tooltip=[
                alt.Tooltip(
                    "completionMinutes",
                    type="quantitative",
                    bin=alt.Bin(step=5),
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Other Statistics
    """)
    return


@app.cell
def _(surveyAnswers):
    def get_perc_values(questionID: str):
        answers = surveyAnswers[surveyAnswers.questionID == questionID]
        n = len(answers)
        a = None if n == 0 else (len(answers[answers.answer_1 == "A"]) / n)
        b = None if n == 0 else (len(answers[answers.answer_1 == "B"]) / n)
        instruments_on = None if n == 0 else (len(answers[answers.backgroundMusic]) / n)
        agreement = None if n == 0 else max(a, b)
        return {"num_answers": n, "A_perc": a, "B_perc": b, "agreement": agreement, "instruments_on": instruments_on}

    return (get_perc_values,)


@app.cell
def _(get_perc_values, pd, surveyQuestions):
    surveyQuestions[["num_answers", "A_perc", "B_perc", "agreement", "instruments_on"]] = (
        surveyQuestions.index.to_series().apply(get_perc_values).apply(pd.Series)
    )
    surveyQuestions
    return


@app.cell
def _(surveyQuestions):
    # agreement for multiple answers per question:
    multi_answer_mask = surveyQuestions.num_answers > 1
    pe = 0.5
    po = surveyQuestions[multi_answer_mask].agreement.mean()

    kappa = (po - pe) / (1 - pe)
    print(f"""
    Number of questions with multiple answers: {len(surveyQuestions[multi_answer_mask])}
    Answer agreement (po): {100 * po:.1f}%
    Chance agreement (pe): 50%
    Cohen's Kappa:    {kappa:.3f}
    """)
    return multi_answer_mask, pe


@app.cell
def _(multi_answer_mask, surveyQuestions):
    instrument_mask1 = surveyQuestions.instruments_on > 0
    instrument_mask2 = surveyQuestions.instruments_on < 1
    surveyQuestions[instrument_mask1 & instrument_mask2 & multi_answer_mask]
    return instrument_mask1, instrument_mask2


@app.cell
def _(
    instrument_mask1,
    instrument_mask2,
    multi_answer_mask,
    pe,
    surveyQuestions,
):
    po_i_differ = surveyQuestions[instrument_mask1 & instrument_mask2 & multi_answer_mask].agreement.mean()

    kappa_i_differ = (po_i_differ - pe) / (1 - pe)
    print(f"""
    Different Instrument Settings:
    Number of multiple answers per question with different instrument settings: {len(surveyQuestions[instrument_mask1 & instrument_mask2 & multi_answer_mask])}
    Answer agreement (po): {100 * po_i_differ:.1f}%
    Cohen's Kappa:    {kappa_i_differ:.3f}
    """)
    return


@app.cell
def _(
    instrument_mask1,
    instrument_mask2,
    multi_answer_mask,
    pe,
    surveyQuestions,
):
    po_i_same = surveyQuestions[~(instrument_mask1 & instrument_mask2) & multi_answer_mask].agreement.mean()

    kappa_i_same = (po_i_same - pe) / (1 - pe)
    print(f"""
    Same Instrument Settings:
    Number of multiple answers per question with same instrument settings: {len(surveyQuestions[~(instrument_mask1 & instrument_mask2) & multi_answer_mask])}
    Answer agreement (po): {100 * po_i_same:.1f}%
    Cohen's Kappa:    {kappa_i_same:.3f}
    """)
    return


if __name__ == "__main__":
    app.run()
