import os

import numpy as np
import pandas as pd

# -- consts --
GENDER_M_THRESHOLD = 0.4
GENDER_F_THRESHOLD = 1 - GENDER_M_THRESHOLD


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


def calc_gmsi_active_engagement_value(row):
    """Calcualates the GoldMSI value for a participant row

    Args:
        row: the participant row

    Returns:
        float: the estimated active engagement GoldMSI value
    """
    try:
        reverse_scores = [7, 6, 5, 4, 3, 2, 1]
        reverse_score: float = lambda x: reverse_scores[int(x) - 1]
        active_engagement = (
            row["gmsi1"]
            + row["gmsi2"]
            + row["gmsi3"]
            + row["gmsi4"]
            + reverse_score(row["gmsi5"])
            + row["gmsi6"]
            + row["gmsi7"]
        )
        return active_engagement / 7
    except:
        return None


def get_gender_distribution(row, track_df):
    pred_p_X = track_df.loc[row["X"]]["pred_p_male"]
    pred_p_A = track_df.loc[row["A"]]["pred_p_male"]
    pred_p_B = track_df.loc[row["B"]]["pred_p_male"]

    if pred_p_X >= GENDER_M_THRESHOLD and pred_p_A >= GENDER_M_THRESHOLD and pred_p_B >= GENDER_M_THRESHOLD:
        return 1.0  # only male
    elif pred_p_X <= GENDER_F_THRESHOLD and pred_p_A <= GENDER_F_THRESHOLD and pred_p_B <= GENDER_F_THRESHOLD:
        return 0.0  # only female
    elif pred_p_A >= GENDER_M_THRESHOLD and pred_p_B >= GENDER_M_THRESHOLD:
        return 0.75  # only male references, female reference
    elif pred_p_A <= GENDER_F_THRESHOLD and pred_p_B <= GENDER_F_THRESHOLD:
        return 0.25  # only female references, male reference
    else:
        return 0.5  # mixed reference voices


# -- dataset converters --

def load_answers_df(csv_path: str):
    """Create the survey answers df from the survey results

    Args:
        csv_path (str): Path to the survey csv

    Returns:
        pd.DataFrame: The survey answers dataframe
    """
    surveyAnswers = pd.read_csv(csv_path, index_col="_id")
    surveyAnswers["editDate"] = parse_js_date(surveyAnswers["editDate"])
    surveyAnswers["createDate"] = parse_js_date(surveyAnswers["createDate"])
    return surveyAnswers

def load_questions_df(csv_path: str, answers_df):
    """Create the survey questions df from the survey results

    Args:
        csv_path (str): Path to the survey csv
        answers_df (pd.DataFrame): The answers data frame

    Returns:
        pd.DataFrame: The survey questions dataframe
    """
    surveyQuestions = pd.read_csv(csv_path, index_col="_id")
    surveyQuestions["randomized"] = surveyQuestions.questionnaireID < 3
    surveyQuestions[["num_answers", "A_perc", "B_perc", "agreement", "instruments_on"]] = (
        surveyQuestions.index.to_series().apply(lambda x: get_answer_ratios(x, answers_df))
    )
    surveyQuestions["editDate"] = parse_js_date(surveyQuestions["editDate"])
    surveyQuestions["createDate"] = parse_js_date(surveyQuestions["createDate"])
    surveyQuestions.dropna(axis=0, inplace=True)
    return surveyQuestions


def load_participant_df(csv_path: str, answers_df: pd.DataFrame):
    """Create the participant df from the survey results

    Args:
        csv_path (str): Path to the survey csv
        answers_df (pd.DataFrame): The answers data frame

    Returns:
        pd.DataFrame: The participant dataframe
    """
    participants = pd.read_csv(csv_path, index_col="_id")
    participants["editDate"] = parse_js_date(participants["editDate"])
    participants["createDate"] = parse_js_date(participants["createDate"])
    start_time = participants["createDate"]
    end_time = participants.index.to_series().apply(
        lambda x: answers_df[answers_df.participantID == x].editDate.max()
    )
    participants["completionTime"] = end_time - start_time 
    participants["completionMinutes"] = participants["completionTime"].dt.total_seconds() / 60
    participants["gmsi_active_engagement"] = participants.apply(calc_gmsi_active_engagement_value, axis=1)
    return participants # [participants.surveyCompleted]


def load_survey_data(csv_paths: dict):
    """Loads the survey data from the given paths

    Args:
        csv_paths (dict): a dict with paths to the csv files, tracks csv is optional

    Returns:
        dict | tuple: a dict containing the dataframes and additional statistical dataset information
    """
    survey_data = {}

    survey_data["songs_df"] = pd.read_csv(csv_paths["songs"], index_col="_id")
    survey_data["answers_df"] = load_answers_df(csv_paths["answers"])
    survey_data["participants_df"] = load_participant_df(csv_paths["participants"], survey_data["answers_df"])
    survey_data["questions_df"] = load_questions_df(csv_paths["questions"], survey_data["answers_df"])

    if csv_paths.get("tracks") is not None:
        survey_data["track_df"] = pd.read_csv(
            csv_paths["tracks"],
            index_col="track_id",
        )
        survey_data["questions_df"]["gender_distribution"] = survey_data["questions_df"].apply(
            lambda x: get_gender_distribution(x, survey_data["track_df"]), axis=1
        )

    # additional post processing of survey answers, has to be
    survey_data["answers_df"] = filter_answers(survey_data["answers_df"], survey_data["questions_df"])

    survey_data["human_agreement"] = get_human_agreement(survey_data["questions_df"])
    survey_data["answer_a_b_ratio"] = get_a_b_answer_ratio(survey_data["answers_df"])

    return survey_data
