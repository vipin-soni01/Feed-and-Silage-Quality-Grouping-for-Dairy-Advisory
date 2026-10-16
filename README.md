# Feed and Silage Quality Grouping for Dairy Advisory

An unsupervised machine learning project that groups silage samples by their measured nutritional and fermentation characteristics and provides a quality-group prediction to support dairy feed advisory.

## 1. Problem Statement

Silage quality influences feed management and dairy production. This project explores whether unsupervised clustering can group silage samples with similar characteristics and provide an interpretable quality-group advisory.

The project aims to:

- Group silage samples using measured nutritional and fermentation features.
- Compare discovered clusters with reference quality groups derived from the Fermentation Quality Index (FQI).
- Evaluate clustering using silhouette score, cluster purity, and Adjusted Rand Index (ARI).
- Compare the clustering approach with a moisture-threshold baseline.
- Provide an interactive Streamlit interface for sample-level predictions.

## 2. Dataset

The project uses the silage dataset available from the University of Padua research data repository.

- **Dataset:** [Silage research dataset](https://researchdata.cab.unipd.it/883/)
- **Research paper:** [Data 2023, 8(7), 117](https://www.mdpi.com/2306-5729/8/7/117)
- **Source file:** `data/data_silage.xlsx`
- **Worksheet:** `elab.all`
- **Original dataset size:** 1,496 rows and 52 columns.

The evaluation uses 1,485 samples with usable FQI-derived reference grades and moisture measurements. The remaining 11 samples are excluded from this evaluation.

### Reference quality groups

For evaluation, FQI is converted into three reference groups:

| FQI range | Reference group |
|---|---|
| FQI < 47.7 | Lower |
| 47.7 ≤ FQI < 57.4 | Good |
| FQI ≥ 57.4 | Excellent |

These are FQI-derived reference categories, not independent laboratory-grade labels.

## 3. Methodology

### Data preprocessing

- Select nutritional and fermentation measurements as clustering features.
- Impute missing feature values using the median.
- Standardize features before applying K-Means.
- Exclude FQI from clustering features to avoid target leakage.
- Exclude identifiers and redundant or unsuitable columns from the feature set.

### Clustering algorithm

The primary model uses K-Means with seven clusters.

The 17 features are:

`dm.s`, `ash.s`, `cp.s`, `ee.s`, `ndf.s`, `adf.s`, `starch.s`, `pH`, `ammonia.s`, `glucose.s`, `fructose.s`, `ethanol.s`, `lactic.ac.s`, `acetic.ac.s`, `propionic.ac.s`, `butyric.ac.s`, and `dm.loss`.

### Held-out evaluation

A reproducible stratified train/test split is used:

- Training samples: 1,113
- Test samples: 372
- Test proportion: 25%
- Random seed: 42

The imputer, scaler, and K-Means model are fitted on training data only. Cluster-to-grade mapping is learned from training labels and then applied to the test clusters.

### Moisture baseline

A single moisture threshold is used as a baseline for distinguishing Excellent from Not Excellent.

- Moisture is calculated as `100 - dm.s`.
- The threshold is selected using training data only.
- The selected threshold for this split is approximately **68.44%**.

This is a data-derived threshold for the current experiment, not a universal agricultural standard.

## 4. Held-Out Results

Results from the current 25% test split:

| Metric | Result |
|---|---:|
| Number of K-Means clusters | 7 |
| Test silhouette score | 0.2897 |
| Test Adjusted Rand Index (ARI) | 0.1398 |
| Test cluster purity | 61.02% |
| Three-class balanced accuracy after cluster mapping | 60.99% |
| Binary balanced accuracy of cluster-mapped predictions | 79.49% |
| Binary balanced accuracy of moisture baseline | 76.46% |

### Interpretation

The silhouette score indicates some cluster structure, with overlap between groups. The ARI indicates limited agreement between discovered clusters and the FQI-derived reference categories.

The binary cluster-mapped predictions achieved 79.49% balanced accuracy, compared with 76.46% for the moisture baseline on the same test samples. This difference is specific to one test split and does not establish that clustering will outperform the baseline on other samples.

The three-class results also show that the Good category is harder to identify than the Lower and Excellent categories.

These findings should be treated as experimental results rather than a validated agricultural grading system.

## 5. Project Structure

```text
Unsupervised Project/
├── app/
│   └── app.py
├── data/
│   └── data_silage.xlsx
├── models/
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── quality_mapping.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/
│   ├── heldout_metrics.json
│   ├── test_cluster_grade_counts.csv
│   └── test_classification_report.csv
├── src/
│   ├── predict.py
│   ├── test_prediction.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## 6. Installation

### Prerequisites

- Python compatible with the pinned dependencies in `requirements.txt`
- Git
- A Windows, Linux, or macOS terminal

### Clone the repository

```bash
git clone https://github.com/vipin-soni01/Feed-and-Silage-Quality-Grouping-for-Dairy-Advisory.git
cd Feed-and-Silage-Quality-Grouping-for-Dairy-Advisory
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 7. Run the Application

From the repository root, run:

```bash
python -m streamlit run app/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the advisory interface.

## 8. Run the Evaluation

From the repository root, with the project environment activated:

```bash
python src/evaluation.py
```

The script writes the evaluation outputs to the `results/` directory:

- `heldout_metrics.json`
- `test_cluster_grade_counts.csv`
- `test_classification_report.csv`

The evaluation script trains its own clustering pipeline for the held-out experiment and does not intentionally overwrite the saved model files used by the application.

## 9. Limitations and Future Work

- FQI-derived reference categories are proxies, not independent laboratory-grade labels.
- Results currently come from one stratified train/test split.
- Cluster purity depends on the cluster-to-grade mapping and should not be interpreted as universal classification accuracy.
- Moisture alone is a simple baseline and cannot represent every aspect of silage quality.
- The current evaluation does not establish that the advisory is suitable for real-world feeding decisions.

Future improvements include:

- Repeated stratified evaluations or cross-validation.
- Testing alternative clustering methods, including hierarchical clustering and DBSCAN.
- Comparing multiple values of K using internal and external metrics.
- Adding visualizations of cluster profiles and feature importance for interpretation.
- Validating advisory outputs with independent laboratory assessments and domain experts.
- Improving input validation and documenting acceptable measurement ranges in the application.

## 10. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Excel dataset processing

## 11. Author and Repository

**GitHub repository:** [Feed and Silage Quality Grouping for Dairy Advisory](https://github.com/vipin-soni01/Feed-and-Silage-Quality-Grouping-for-Dairy-Advisory)

This project was developed as an unsupervised learning and dairy-feed advisory project. Its outputs are intended for research and educational purposes, not as a substitute for professional feed-quality testing or veterinary advice.
