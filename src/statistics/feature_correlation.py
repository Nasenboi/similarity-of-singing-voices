from typing import List, Literal

import marimo as mo
import numpy as np
import pandas as pd
from scipy.spatial.distance import canberra, chebyshev, cosine, euclidean, minkowski
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler

# -- consts --
MINKOWSKI_P = 4
DISTANCE_ALGORITHMS = Literal[
    "euclidean",
    "chebyshev",
    "cosine",
    "minkowski",
    "canberra",
]
DISTANCE_ALGORITHMS_L = [
    "euclidean",
    "chebyshev",
    "cosine",
    "minkowski",
    "canberra",
]


def get_distance_diff(x, a_1, a_2) -> float:
    """gets the difference of distananced between dist(X, A1) and dist(X, A2)

    Args:
        x: feature X value
        a_1: feature A1 value
        a_2:feature A2 value

    Returns:
        float: the normalized distance difference:
               0 => A2 is similar to X, 0.5 => both are equally similar, 1 => A1 is more similar to X
    """
    if pd.api.types.is_numeric_dtype(type(x)):
        dist_a_1 = abs(x - a_1)
        dist_a_2 = abs(x - a_2)
        if dist_a_1 + dist_a_2 == 0:
            return 0.5  # small failsave to avoid zero division
        # -1 to 1: -1 => A2 is more similar to X, 1 => A1 is more similar to X
        norm_dist = (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
        # 0 to 1: 0 => A2 is more similar to X, 1 => A1 is more similar to X
        return (norm_dist + 1) / 2
    else:
        if a_1 == a_2:
            return 0.5  # Uncertainty
        elif a_1 == x:
            return 1.0  # A1 more similar to X
        elif a_2 == x:
            return 0.0  # A2 more similar to X
        else:
            return 0.5  # Uncertainty


def get_local_feature_distance_diffs(
    question,
    feature_df: pd.DataFrame,
    feature_key: str,
) -> float:
    x = feature_df.loc[question["X"]][feature_key]
    a_1 = feature_df.loc[question["A"]][feature_key]
    a_2 = feature_df.loc[question["B"]][feature_key]
    return get_distance_diff(x, a_1, a_2)


def get_all_distance_differences(
    feature_df: pd.DataFrame, feature_list: List[str], questions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the distance differences for all features and all rows

    Args:
        feature_df (pd.DataFrame): the feature dataframe
        feature_list (List[str]): the list of features
        questions_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    columns = {}
    for feat in mo.status.progress_bar(
        feature_list,
        title="Calculating Feature Distance Differences...",
        remove_on_exit=True,
    ):
        feat_scores = questions_df.apply(
            lambda x, f=feat: get_local_feature_distance_diffs(x, feature_df, f),
            axis=1,
        )
        columns[feat] = feat_scores.to_list()
    return pd.DataFrame(columns, index=questions_df.index)


def get_mean_values(distance_score_df: pd.DataFrame, feature_list=None, top_x: int = None) -> dict:
    """get the top x mean values from a distance score df

    Args:
        agreement_df (pd.DataFrame): _description_
        feature_list (_type_, optional): _description_. Defaults to None.
        top_x (int, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    f_iterator = feature_list if feature_list is not None else distance_score_df.columns
    mean_values = {feat: distance_score_df[feat].mean() for feat in f_iterator}
    if top_x is not None:
        mean_values = dict(sorted(mean_values.items(), key=lambda item: item[1], reverse=True)[:top_x])
    return mean_values


def scale_df(feature_df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """scales feature columns using the standard scaler (std_dev = +-1.0)

    Args:
        feature_df (pd.DataFrame): the feature data frame
        columns (List[str], optional): An optional list of columnd to scale. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    scaler = StandardScaler()
    df = feature_df.copy()
    cols = columns if columns is not None else feature_df.columns
    df[cols] = scaler.fit_transform(feature_df[cols])
    return df


def get_distance_row(question, feature_df: pd.DataFrame, distance_algorithm: DISTANCE_ALGORITHMS) -> float:
    """get the global distance value for a single row using a specified distance algorithm

    Args:
        question: the survey question
        feature_df (pd.DataFrame): feature dataframe
        distance_algorithm (DISTANCE_ALGORITHMS): _description_

    Raises:
        NotImplementedError: When the distance algorithm does not exist

    Returns:
        float: The normalized global distance value
    """
    x = feature_df.loc[question["X"]].values
    a_1 = feature_df.loc[question["A"]].values
    a_2 = feature_df.loc[question["B"]].values
    if distance_algorithm == "euclidean":
        dist_a_1 = euclidean(x, a_1)
        dist_a_2 = euclidean(x, a_2)
    elif distance_algorithm == "chebyshev":
        dist_a_1 = chebyshev(x, a_1)
        dist_a_2 = chebyshev(x, a_2)
    elif distance_algorithm == "cosine":
        dist_a_1 = cosine(x, a_1)
        dist_a_2 = cosine(x, a_2)
    elif distance_algorithm == "minkowski":
        dist_a_1 = minkowski(x, a_1, p=MINKOWSKI_P)
        dist_a_2 = minkowski(x, a_2, p=MINKOWSKI_P)
    elif distance_algorithm == "canberra":
        dist_a_1 = canberra(x, a_1)
        dist_a_2 = canberra(x, a_2)
    else:
        raise NotImplementedError(f"Distance Measure {distance_algorithm} is not implemented yet!")
    # -1 to 1: -1 => A2 is more similar to X, 1 => A1 is more similar to X
    norm_dist = (dist_a_2 - dist_a_1) / (dist_a_1 + dist_a_2)
    # 0 to 1: 0 => A2 is more similar to X, 1 => A1 is more similar to X
    return (norm_dist + 1) / 2


def get_global_distance_scores(feature_df: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    """Get global distance scores => global = using all features for a single distance

    Distance algorithms:
    - euclidean: standard euclidean distance, spreads differences across all features
    - chebyshev: max difference across all dimensions, highly sensitive to single outlier features
    - cosine: measures angle between vectors, captures timbral character regardless of magnitude
    - minkowski: generalization of euclidean with stronger outlier amplification
    - canberra: normalizes per-dimension, amplifies subtle differences in low-valued features

        Args:
            feature_df (pd.DataFrame): _description_
            questions_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
    """
    gda_df = pd.DataFrame()
    for d in mo.status.progress_bar(DISTANCE_ALGORITHMS_L, title="Calculating GDAs", remove_on_exit=True):
        gda_df[f"distance_{d}"] = questions_df.apply(
            lambda x: get_distance_row(x, feature_df, d),
            axis=1,
        )
    return gda_df


def get_anova_values(*args) -> tuple:
    f_statistic, p_value = f_oneway(*args)
    group_sizes = {str(i): len(g) for i, g in enumerate(args)}
    print("Group sizes:\n", group_sizes)
    print(f"F-statistic: {f_statistic:0.3f}")
    print(f"P-value: {p_value:0.3f}")
    return f_statistic, p_value
