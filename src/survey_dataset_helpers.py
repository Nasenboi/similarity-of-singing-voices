import os

import numpy as np
import pandas as pd


# -- basic helper functions --
def parse_js_date(series: pd.Series):
    """Parses a JavaScript date into the pandas date format

    Args:
        series (pd.Series): The column to parse (should be string values)

    Returns:
        pd.Series: The converted series
    """
    cleaned = series.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    return pd.to_datetime(cleaned, format="%a %b %d %Y %H:%M:%S GMT%z")


def get_track_ids_for_answer(row, questions_df: pd.DataFrame):
    """Fetches the track ids for the answer

    Args:
        row: _description_
        questions_df (pd.DataFrame): the survey question dataframe

    Returns:
        pd.Series: The track ids for the answer
    """
    questionnaireID = row["questionID"]
    answerKey1 = row["answer_1"]
    answerKey2 = row["answer_2"]
    question = questions_df.loc[questionnaireID]
    return pd.Series(
        {
            "track_id_X": question["X"],
            "track_id_1": question[answerKey1],
            "track_id_2": question[answerKey2],
            "skipped": question["skip"],
        }
    )


def filter_answers(answers_df: pd.DataFrame, questions_df: pd.DataFrame):
    """Creates new columns for the answers dataframes and filters out unused ones

    Args:
        answers_df (pd.DataFrame): the survey answers df
        questions_df (pd.DataFrame): the survey questions df

    Returns:
        pd.DataFrame: the new survey answers df
    """
    answers_df[["track_id_X", "track_id_1", "track_id_2", "skipped"]] = answers_df.apply(
        lambda x: get_track_ids_for_answer(x, questions_df), axis=1
    )
    return answers_df[~answers_df.skipped].drop(columns="skipped")


def get_answer_ratios(questionID: str, answers_df: pd.DataFrame):
    """Gets the A/B answer ratio in percent for a question

    Args:
        questionID (str): The question id
        answers_df (pd.DataFrame): the survey answers

    Returns:
        pd.Series: The answer ratios
    """
    answers = answers_df[answers_df.questionID == questionID]
    n = len(answers)
    a = None if n == 0 else (len(answers[answers.answer_1 == "A"]) / n)
    b = None if n == 0 else (len(answers[answers.answer_1 == "B"]) / n)
    instruments_on = None if n == 0 else (len(answers[answers.backgroundMusic]) / n)
    agreement = None if n == 0 else max(a, b)
    return pd.Series(
        {
            "num_answers": n,
            "A_perc": a,
            "B_perc": b,
            "agreement": agreement,
            "instruments_on": instruments_on,
        }
    )


def get_human_agreement(questions_df: pd.DataFrame):
    """Returns the overall agreement of human answers

    Args:
        questions_df (pd.DataFrame): the survey questions df

    Returns:
        float: the overall mean agreement of human answers in percent
    """
    multi_answer_mask = questions_df.num_answers > 1
    return questions_df[multi_answer_mask].agreement.mean()


def get_a_b_answer_ratio(answers_df: pd.DataFrame):
    """Returns the overall ratio of A/B answers as the random baseline, should be close to 50%

    Args:
        answers_df (pd.DataFrame): the survey answers df

    Returns:
        float: the answer A/B ratio in percent (1 only B, 0 only A)
    """
    return len(answers_df[answers_df.answer_1 == "B"]) / len(answers_df)


# -- dataset converters --


def load_questions_df(csv_path: str, answers_df):
    """Create the survey questions df from the survey results

    Args:
        csv_path (str): Path to the survey csv

    Returns:
        pd.DataFrame: The survey questions dataframe
    """
    surveyQuestions = pd.read_csv(csv_path, index_col="_id")
    surveyQuestions["randomized"] = surveyQuestions.questionnaireID < 3
    surveyQuestions[["num_answers", "A_perc", "B_perc", "agreement", "instruments_on"]] = (
        surveyQuestions.index.to_series().apply(lambda x: get_answer_ratios(x, answers_df))
    )
    surveyQuestions.dropna(axis=0, inplace=True)
    return surveyQuestions


def load_participant_df(csv_path: str):
    """Create the participant df from the survey results

    Args:
        csv_path (str): Path to the survey csv

    Returns:
        pd.DataFrame: The participant dataframe
    """
    participants = pd.read_csv(csv_path, index_col="_id")
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    participants[participants.surveyCompleted]
    participants["completionTime"] = participants["editDate"] - participants["createDate"]
    participants["completionMinutes"] = participants["completionTime"].dt.total_seconds() / 60
    return participants


def load_survey_data(csv_paths: dict):
    """Loads the survey data from the given paths

    Args:
        csv_paths (dict): a dict with paths to the csv files

    Returns:
        dict | tuple: a dict containing the dataframes and additional statistical dataset information
    """
    survey_data = {}

    survey_data["songs_df"] = pd.read_csv(csv_paths["songs"], index_col="_id")
    survey_data["answers_df"] = pd.read_csv(csv_paths["answers"], index_col="_id")
    survey_data["participants_df"] = load_participant_df(csv_paths["participants"])
    survey_data["questions_df"] = load_questions_df(csv_paths["questions"], survey_data["answers_df"])

    # additional post processing of survey answers, has to be
    survey_data["answers_df"] = filter_answers(survey_data["answers_df"], survey_data["questions_df"])

    survey_data["human_agreement"] = get_human_agreement(survey_data["questions_df"])
    survey_data["answer_a_b_ratio"] = get_a_b_answer_ratio(survey_data["answers_df"])

    return survey_data
