import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data_silage.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = [
    "dm.s", "ash.s", "cp.s", "ee.s", "ndf.s", "adf.s",
    "starch.s", "pH", "ammonia.s", "glucose.s", "fructose.s",
    "ethanol.s", "lactic.ac.s", "acetic.ac.s", "propionic.ac.s",
    "butyric.ac.s", "dm.loss",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25
K = 7
FQI_BINS = [-np.inf, 47.7, 57.4, np.inf]
GRADES = ["Lower", "Good", "Excellent"]


def main():
    data = pd.read_excel(DATA_PATH, sheet_name="elab.all")

    data["reference_grade"] = pd.cut(
        data["fqi"],
        bins=FQI_BINS,
        labels=GRADES,
        right=False,
    )

    data["moisture"] = 100 - data["dm.s"]

    # Keep rows with a reference grade and moisture for fair comparison.
    valid = data["reference_grade"].notna() & data["moisture"].notna()
    data = data.loc[valid].copy()

    X = data[FEATURES]
    y = data["reference_grade"].astype(str)
    moisture = data["moisture"]

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    moisture_train = moisture.iloc[train_idx]
    moisture_test = moisture.iloc[test_idx]

    # Fit every preprocessing/model step on training data only.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
    train_clusters = model.fit_predict(X_train_scaled)
    test_clusters = model.predict(X_test_scaled)

    # Learn cluster-to-grade mapping using training labels only.
    train_table = pd.crosstab(
        pd.Series(train_clusters, name="cluster"),
        pd.Series(y_train.to_numpy(), name="grade"),
    ).reindex(columns=GRADES, fill_value=0)

    cluster_mapping = {
        int(cluster): str(train_table.loc[cluster].idxmax())
        for cluster in train_table.index
    }
    test_predictions = np.array(
        [cluster_mapping[int(cluster)] for cluster in test_clusters]
    )

    # Three-class balanced accuracy for cluster-mapped predictions.
    cluster_balanced_accuracy = balanced_accuracy_score(
        y_test, test_predictions
    )

    # Fair binary comparison: Excellent vs. Not Excellent.
    cluster_binary_test = (test_predictions == "Excellent").astype(int)
    cluster_binary_balanced_accuracy = balanced_accuracy_score(
        y_test_binary if "y_test_binary" in locals()
        else (y_test == "Excellent").astype(int),
        cluster_binary_test,
    )

    # Test-only clustering metrics.
    test_silhouette = silhouette_score(X_test_scaled, test_clusters)
    test_ari = adjusted_rand_score(y_test, test_clusters)

    # Test-only mapped cluster purity.
    test_table = pd.crosstab(
        pd.Series(test_clusters, name="cluster"),
        pd.Series(y_test.to_numpy(), name="grade"),
    ).reindex(columns=GRADES, fill_value=0)

    purity = float(test_table.max(axis=1).sum() / test_table.to_numpy().sum())

    # Choose a moisture threshold on training data only.
    # Objective: balanced accuracy for Excellent vs Not Excellent.
    y_train_binary = (y_train == "Excellent").astype(int)
    y_test_binary = (y_test == "Excellent").astype(int)

    candidate_thresholds = np.unique(
        np.quantile(moisture_train.dropna(), np.linspace(0, 1, 501))
    )

    best_threshold = None
    best_score = -1.0

    for threshold in candidate_thresholds:
        train_baseline = (moisture_train >= threshold).astype(int)
        score = balanced_accuracy_score(y_train_binary, train_baseline)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    test_baseline = np.where(
        moisture_test.to_numpy() >= best_threshold, 1, 0
    )
    baseline_test_score = balanced_accuracy_score(
        y_test_binary, test_baseline
    )

    results = {
        "dataset_rows_evaluated": int(len(data)),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "kmeans_clusters": K,
        "test_silhouette": float(test_silhouette),
        "test_ari_vs_fqi_grades": float(test_ari),
        "test_cluster_purity": purity,
        "test_cluster_balanced_accuracy": float(cluster_balanced_accuracy),
        "test_cluster_binary_balanced_accuracy": float(cluster_binary_balanced_accuracy),
        "cluster_to_grade_mapping_learned_on_train": cluster_mapping,
        "moisture_threshold_learned_on_train": best_threshold,
        "moisture_baseline_test_balanced_accuracy": float(baseline_test_score),
        "moisture_baseline_train_balanced_accuracy": float(best_score),
    }

    with open(
        os.path.join(RESULTS_DIR, "heldout_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    test_table.to_csv(os.path.join(RESULTS_DIR, "test_cluster_grade_counts.csv"))

    report = classification_report(
        y_test,
        test_predictions,
        labels=GRADES,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(RESULTS_DIR, "test_classification_report.csv")
    )

    print("\n=== HELD-OUT EVALUATION RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\nTest cluster vs. reference-grade counts:")
    print(test_table)
    print("\nClassification report for train-mapped cluster labels:")
    print(classification_report(y_test, test_predictions, labels=GRADES, zero_division=0))
    print(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
