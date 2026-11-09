import pandas as pd
import numpy as np


def calculate_NEWS2(df, columns):
    """
    Calculate the original National Early Warning Score (NEWS, 2012).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing patient observations.

    columns : list
        List of 7 column names, in this exact order:

        0. Respiratory rate (breaths/min)
        1. Oxygen saturation (SpO2, %)
        2. Supplemental oxygen (0 = No, 1 = Yes)
        3. Systolic blood pressure (mmHg)
        4. Heart rate (beats/min)
        5. Consciousness (0 = Alert, 1 = V/P/U)
        6. Temperature (°C)

    Returns
    -------
    pandas.DataFrame
        Copy of the original dataset with a new 'NEWS' column.
        Missing or invalid measurements produce NaN.
    """

    if len(columns) != 7:
        raise ValueError("Exactly 7 column names are required.")

    if len(set(columns)) != 7:
        raise ValueError("Column names must be unique.")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = df.copy()

    rr, spo2, oxygen, sbp, hr, consciousness, temp = [
        pd.to_numeric(result[col], errors="coerce")
        for col in columns
    ]

    # Respiratory rate
    rr_score = np.select(
        [
            rr <= 8,
            (rr >= 9) & (rr <= 11),
            (rr >= 12) & (rr <= 20),
            (rr >= 21) & (rr <= 24),
            rr >= 25
        ],
        [3, 1, 0, 2, 3],
        default=np.nan
    )

    # Oxygen saturation (SpO2)
    spo2_score = np.select(
        [
            spo2 <= 91,
            (spo2 > 91) & (spo2 < 94),
            (spo2 >= 94) & (spo2 <= 95),
            spo2 >= 96
        ],
        [3, 2, 1, 0],
        default=np.nan
    )

    # Supplemental oxygen
    oxygen_score = np.select(
        [oxygen == 0, oxygen == 1],
        [0, 2],
        default=np.nan
    )

    # Systolic blood pressure
    sbp_score = np.select(
        [
            sbp <= 90,
            (sbp >= 91) & (sbp <= 100),
            (sbp >= 101) & (sbp <= 110),
            (sbp >= 111) & (sbp <= 219),
            sbp >= 220
        ],
        [3, 2, 1, 0, 3],
        default=np.nan
    )

    # Heart rate
    hr_score = np.select(
        [
            hr <= 40,
            (hr >= 41) & (hr <= 50),
            (hr >= 51) & (hr <= 90),
            (hr >= 91) & (hr <= 110),
            (hr >= 111) & (hr <= 130),
            hr >= 131
        ],
        [3, 1, 0, 1, 2, 3],
        default=np.nan
    )

    # Consciousness (AVPU)
    consciousness_score = np.select(
        [consciousness == 1, consciousness == 0],
        [0, 3],
        default=np.nan
    )

    # Temperature
    temp_score = np.select(
        [
            temp <= 35.0,
            (temp > 35.0) & (temp <= 36.0),
            (temp > 36.0) & (temp <= 38.0),
            (temp > 38.0) & (temp <= 39.0),
            temp > 39.0
        ],
        [3, 1, 0, 1, 2],
        default=np.nan
    )

    scores = np.column_stack([
        rr_score,
        spo2_score,
        oxygen_score,
        sbp_score,
        hr_score,
        consciousness_score,
        temp_score
    ])

    # Do not calculate NEWS if any required measurement is missing.
    result["NEWS2"] = np.where(
        np.isnan(scores).any(axis=1),
        np.nan,
        scores.sum(axis=1)
    )

    return result