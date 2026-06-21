import marimo

__generated_with = "0.23.10"
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
    from src.globals import (
        AUDIO_FOLDER,
        CSV_FOLDER,
        DATASET_FOLDER,
        STEMS_FOLDER,
        TRACKS_PATH,
        UVR_MODEL_PATH,
    )
    from src.survey_dataset_helpers import load_survey_data

    return DATASET_FOLDER, load_survey_data, mo, os


@app.cell
def _(DATASET_FOLDER, os):
    SURVEY_FOLDER = os.path.join(DATASET_FOLDER, "survey", "survey_2")
    CSV_PATHS = {
        "participants": os.path.join(SURVEY_FOLDER, "participants.csv"),
        "songs": os.path.join(SURVEY_FOLDER, "songs.csv"),
        "answers": os.path.join(SURVEY_FOLDER, "surveyAnswers.csv"),
        "questions": os.path.join(SURVEY_FOLDER, "surveyQuestions.csv"),
    }
    return (CSV_PATHS,)


@app.cell
def _(mo):
    mo.md(r"""
    # Load all Datasets
    """)
    return


@app.cell
def _(CSV_PATHS, load_survey_data):
    SURVEY_DATA = load_survey_data(CSV_PATHS)
    questions_df = SURVEY_DATA["questions_df"]
    answers_df = SURVEY_DATA["answers_df"]
    participants_df = SURVEY_DATA["participants_df"]
    songs_df = SURVEY_DATA["songs_df"]
    human_agreement = SURVEY_DATA["human_agreement"]
    answer_a_b_ratio = SURVEY_DATA["answer_a_b_ratio"]
    return answers_df, participants_df, questions_df


@app.cell
def _(questions_df):
    # There should be no duplicate triplets in all questions_df
    duplicates = questions_df.duplicated(subset=["X", "A", "B"], keep=False)
    print(questions_df[duplicates])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Initial Statistics
    """)
    return


@app.cell
def _(answers_df, participants_df, questions_df):
    ab_ratio_a = (
        100 * len(answers_df[answers_df.answer_1 == "A"]) / len(answers_df)
    )
    ab_ratio_b = (
        100 * len(answers_df[answers_df.answer_1 == "B"]) / len(answers_df)
    )

    instruments_on_yes = (
        100 * len(answers_df[answers_df.backgroundMusic]) / len(answers_df)
    )
    instruments_on_no = (
        100 * len(answers_df[~answers_df.backgroundMusic]) / len(answers_df)
    )

    print(f"""
    Total number of participants: {len(participants_df)}
    Number of completed surveys:  {
        len(participants_df[participants_df.surveyCompleted])
    }
    Number of incomple surveys:   {
        len(participants_df[~participants_df.surveyCompleted])
    }

    Median completion time: {
        round(participants_df.completionMinutes.median())
    } minutes

    Total number of questions: {len(questions_df)} 
    Number of questions with no answers: {
        questions_df[~questions_df.index.isin(answers_df["questionID"])].shape[0]
    }
    Number of questions with multiple answers: {
        answers_df.groupby("questionID").size().loc[lambda x: x > 1].shape[0]
    }

    Total number of answers: {len(answers_df)}
    Answer A/B: {ab_ratio_a:.1f}% A; {ab_ratio_b:.1f}% B
    Instruments on:   {instruments_on_yes:.1f}% Yes; {instruments_on_no:.1f}% No
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Other Statistics
    """)
    return


@app.cell
def _(questions_df):
    # agreement for multiple answers per question:
    multi_answer_mask = questions_df.num_answers > 1
    pe = 0.5
    po = questions_df[multi_answer_mask].agreement.mean()

    kappa = (po - pe) / (1 - pe)
    print(f"""
    Number of questions with multiple answers: {len(questions_df[multi_answer_mask])}
    Answer agreement (po): {100 * po:.1f}%
    Chance agreement (pe): {100 * pe:.1f}
    Cohen's Kappa:    {kappa:.3f}
    """)
    return multi_answer_mask, pe, po


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Survey Types
    """)
    return


@app.cell
def _(multi_answer_mask, pe, po, questions_df):
    # agreement for max entropy questions:
    randomized_mask = questions_df["randomized"]
    po_max_kappa = questions_df[
        multi_answer_mask & (~randomized_mask)
    ].agreement.mean()

    kappa_max_ent = (po - pe) / (1 - pe)
    print(f"""
    Number of questions of max entropy type: {len(questions_df[multi_answer_mask & (~randomized_mask)])}
    Answer agreement (po): {100 * po_max_kappa:.1f}%
    Chance agreement (pe): {100 * pe:.1f}
    Cohen's Kappa:    {kappa_max_ent:.3f}
    """)
    return (randomized_mask,)


@app.cell
def _(multi_answer_mask, pe, po, questions_df, randomized_mask):
    # agreement for max entropy questions:
    po_rand = questions_df[multi_answer_mask & randomized_mask].agreement.mean()

    kappa_rand = (po - pe) / (1 - pe)
    print(f"""
    Number of questions of max entropy type: {len(questions_df[multi_answer_mask & randomized_mask])}
    Answer agreement (po): {100 * po_rand:.1f}%
    Chance agreement (pe): {100 * pe:.1f}
    Cohen's Kappa:    {kappa_rand:.3f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instruments
    """)
    return


@app.cell
def _(multi_answer_mask, questions_df):
    instrument_mask1 = questions_df.instruments_on > 0
    instrument_mask2 = questions_df.instruments_on < 1
    questions_df[instrument_mask1 & instrument_mask2 & multi_answer_mask]
    return instrument_mask1, instrument_mask2


@app.cell
def _(instrument_mask1, instrument_mask2, multi_answer_mask, pe, questions_df):
    po_i_differ = questions_df[
        instrument_mask1 & instrument_mask2 & multi_answer_mask
    ].agreement.mean()

    kappa_i_differ = (po_i_differ - pe) / (1 - pe)
    print(f"""
    Different Instrument Settings:
    Number of multiple answers per question with different instrument settings: {len(questions_df[instrument_mask1 & instrument_mask2 & multi_answer_mask])}
    Answer agreement (po): {100 * po_i_differ:.1f}%
    Cohen's Kappa:    {kappa_i_differ:.3f}
    """)
    return


@app.cell
def _(instrument_mask1, instrument_mask2, multi_answer_mask, pe, questions_df):
    po_i_same = questions_df[
        ~(instrument_mask1 & instrument_mask2) & multi_answer_mask
    ].agreement.mean()

    kappa_i_same = (po_i_same - pe) / (1 - pe)
    print(f"""
    Same Instrument Settings:
    Number of multiple answers per question with same instrument settings: {len(questions_df[~(instrument_mask1 & instrument_mask2) & multi_answer_mask])}
    Answer agreement (po): {100 * po_i_same:.1f}%
    Cohen's Kappa:    {kappa_i_same:.3f}
    """)
    return


if __name__ == "__main__":
    app.run()
