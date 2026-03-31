"""
Student Performance Dataset Generator
======================================
Generates a synthetic dataset of 250 student records with 9 features
and a binary target (Pass / Fail). The underlying logic embeds realistic
patterns — e.g. high study hours + high attendance correlate with passing —
plus Gaussian noise so the classification boundary is imperfect.

Usage:
    python generate_dataset.py
"""

import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

N_SAMPLES = 250
OUTPUT_PATH = "student_performance.csv"


def generate_features(n: int) -> dict:
    """Generate realistic student feature distributions."""

    study_hours = np.clip(np.random.exponential(2.5, n), 0.5, 10.0)
    attendance = np.clip(np.random.normal(75, 15, n), 30.0, 100.0)
    prev_gpa = np.clip(np.random.normal(7.0, 1.5, n), 0.0, 10.0)
    sleep_hours = np.clip(np.random.normal(7.0, 1.0, n), 3.0, 10.0)
    extracurricular = np.random.binomial(1, 0.4, n)
    internet_hours = np.clip(np.random.exponential(2.0, n), 0.0, 8.0)
    parental_ed = np.random.choice(
        [1, 2, 3, 4], n, p=[0.20, 0.35, 0.30, 0.15]
    )
    absences = np.clip(np.random.poisson(5, n), 0, 30)
    assignment_rate = np.clip(np.random.normal(70, 20, n), 0.0, 100.0)

    return {
        "study_hours_per_day": np.round(study_hours, 1),
        "attendance_percentage": np.round(attendance, 1),
        "previous_gpa": np.round(prev_gpa, 2),
        "sleep_hours_per_day": np.round(sleep_hours, 1),
        "extracurricular_activities": extracurricular,
        "internet_usage_hours": np.round(internet_hours, 1),
        "parental_education_level": parental_ed,
        "number_of_absences": absences,
        "assignment_completion_rate": np.round(assignment_rate, 1),
    }


def compute_labels(features: dict) -> np.ndarray:
    """
    Weighted scoring function that maps features to a latent score.
    Positive weights → help pass; negative weights → push toward fail.
    Gaussian noise is added so the boundary is fuzzy (~62-68 % pass rate).
    """
    score = (
        features["study_hours_per_day"] * 1.50
        + features["attendance_percentage"] * 0.020
        + features["previous_gpa"] * 0.60
        + features["assignment_completion_rate"] * 0.015
        + features["sleep_hours_per_day"] * 0.15
        + features["extracurricular_activities"] * 0.50
        + features["parental_education_level"] * 0.40
        - features["number_of_absences"] * 0.60
        - features["internet_usage_hours"] * 0.40
    )

    noise = np.random.normal(0, 2.0, len(score))
    return (score + noise > 8.0).astype(int)


def main():
    features = generate_features(N_SAMPLES)
    labels = compute_labels(features)

    df = pd.DataFrame(features)
    df["performance"] = pd.Series(labels).map({0: "Fail", 1: "Pass"})

    df.to_csv(OUTPUT_PATH, index=False)

    n_pass = (df["performance"] == "Pass").sum()
    n_fail = (df["performance"] == "Fail").sum()
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"  Total samples : {len(df)}")
    print(f"  Pass          : {n_pass}  ({n_pass / len(df) * 100:.1f} %)")
    print(f"  Fail          : {n_fail}  ({n_fail / len(df) * 100:.1f} %)")
    print(f"  Features      : {len(features)}")
    print(f"  Columns       : {list(df.columns)}")


if __name__ == "__main__":
    main()
