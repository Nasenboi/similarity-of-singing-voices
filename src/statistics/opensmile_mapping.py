"""
Map open smile features to high level voice quality measures

The correlation is given by the following paper:
Memon: [], S. A. (2020). Acoustic Correlates of the Voice Qualifiers: A Survey. https://arxiv.org/abs/2010.15869

# not found:
"pcm_fftMag_mfcc_sma[0]_amean",
"pcm_fftMag_mfcc_sma[1]_amean",
"pcm_fftMag_mfcc_sma[2]_amean",
"pcm_fftMag_mfcc_sma[3]_amean",
"F0final_sma_peakDistStddev",
"""

import pandas as pd

# Features wit
AMBIGUOUS_FEATURES = {
    "voicingFinalUnclipped": "compare",
    "F0semitoneFrom27.5Hz": "gemaps",
    "pcm_RMSenergy_sma": "compare",
}

FEATURE_MAP = {
    "Aphonic": ["F0final_sma_amean", "logHNR_sma_amean"],
    "Biphonic": ["F0final_sma_amean", "F1frequency_sma3nz_amean"],
    "Breathy": [
        "jitterLocal_sma_amean",
        "logHNR_sma_amean",
        "F0semitoneFrom27.5Hz",
        "logRelF0-H1-H2_sma3nz_amean",
        "voicingFinalUnclipped_sma_amean",
        "pcm_RMSenergy_sma",
    ],
    "Covered": [
        "pcm_fftMag_spectralEntropy_sma_amean",
        "F1frequency_sma3nz_amean",
        "F2frequency_sma3nz_amean",
        "F3frequency_sma3nz_amean",
        "F0final_sma_amean",
    ],
    "Creaky": [
        "F0final_sma_amean",
        "jitterLocal_sma_amean",
        "logRelF0-H1-H2_sma3nz_amean",
        "logRelF0-H1-A3_sma3nz_amean",
    ],
    "Diplophonic": ["F0final_sma_amean"],
    "Flutter": ["jitterLocal_sma_amean", "shimmerLocal_sma_amean"],
    "Glottalized": [
        "F1amplitudeLogRelF0_sma3nz_amean",
        "pcm_fftMag_spectralSlope_sma_amean",
    ],
    "Raspy/Hoarse": ["logHNR_sma_amean", "jitterLocal_sma_amean", "shimmerLocal_sma_amean"],
    "Nasal/Honky": [
        "F1amplitudeLogRelF0_sma3nz_amean",
        "F2amplitudeLogRelF0_sma3nz_amean",
        "F3amplitudeLogRelF0_sma3nz_amean",
    ],
    "Jitter": ["jitterLocal_sma_amean"],
    "Pressed": ["F0final_sma_amean", "logRelF0-H1-H2_sma3nz_amean"],
    "Pulsed/VocalFry": ["logHNR_sma_amean", "F0final_sma_amean", "logRelF0-H1-H2_sma3nz_amean"],
    "Resonant": ["F1frequency_sma3nz_amean", "F2frequency_sma3nz_amean", "F3frequency_sma3nz_amean"],
    "Rough": [
        "logHNR_sma_amean",
        "audSpec_Rfilt_sma[0]_flatness",
        "audSpec_Rfilt_sma[1]_flatness",
        "audSpec_Rfilt_sma[2]_flatness",
        "audSpec_Rfilt_sma[3]_flatness",
        "audSpec_Rfilt_sma[4]_flatness",
        "audSpec_Rfilt_sma[5]_flatness",
        "audSpec_Rfilt_sma[6]_flatness",
        "audSpec_Rfilt_sma[7]_flatness",
        "audSpec_Rfilt_sma[8]_flatness",
        "audSpec_Rfilt_sma[9]_flatness",
        "audSpec_Rfilt_sma[10]_flatness",
        "audSpec_Rfilt_sma[11]_flatness",
        "audSpec_Rfilt_sma[12]_flatness",
        "audSpec_Rfilt_sma[13]_flatness",
        "audSpec_Rfilt_sma[14]_flatness",
    ],
    "Shimmer": ["shimmerLocal_sma_amean"],
    "Strained": ["pcm_fftMag_spectralSkewness_sma_amean", "pcm_RMSenergy_sma"],
    "Tremerous": ["jitterLocal_sma_amean", "shimmerLocal_sma_amean"],
    "Twangy": [
        "F1frequency_sma3nz_amean",
        "F2frequency_sma3nz_amean",
        "F1bandwidth_sma3nz_amean",
        "F2bandwidth_sma3nz_amean",
    ],
    "Ventricular/Harsh": ["F0final_sma_range", "jitterLocal_sma_amean"],
    "Wobble": ["jitterLocal_sma_amean", "shimmerLocal_sma_amean"],
    "Yawny": [
        "F1frequency_sma3nz_amean",
        "F2frequency_sma3nz_amean",
        "F1bandwidth_sma3nz_amean",
        "F2bandwidth_sma3nz_amean",
    ],
}


def convert_to_voice_quality_features(gemaps_df: pd.DataFrame, compare_df: pd.DataFrame) -> pd.DataFrame:
    gemaps_features = list(gemaps_df.columns)
    compare_features = list(compare_df.columns)

    columns = {}
    failed_feats = []
    for vq_feat, feats in FEATURE_MAP.items():
        for feat in feats:
            try:
                if feat in AMBIGUOUS_FEATURES.keys():
                    fset = AMBIGUOUS_FEATURES[feat]
                    if fset == "gemaps":
                        sub_features = [f for f in compare_features if feat in f]
                        for f in sub_features:
                            columns[(vq_feat, f)] = gemaps_df[feat]
                    else:
                        sub_features = [f for f in gemaps_features if feat in f]
                        for f in sub_features:
                            columns[(vq_feat, f)] = compare_df[feat]
                elif feat in gemaps_features:
                    columns[(vq_feat, feat)] = gemaps_df[feat]
                else:
                    columns[(vq_feat, feat)] = compare_df[feat]
            except:
                failed_feats.append(feat)

    voice_quality_df = pd.DataFrame(columns, index=gemaps_df.index)
    # print(failed_feats)
    return voice_quality_df
