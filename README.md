# Healthcare Patient Clustering by Clinical Indicators

Unsupervised segmentation of CDC BRFSS 2015 respondents into clinical risk tiers, plus a
Streamlit app for exploring the segments and assigning a new patient to one.

**Dataset:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(Kaggle, `alexteboul`) — 253,680 survey respondents, 21 health indicators.
We use `diabetes_012_health_indicators_BRFSS2015.csv`. The diabetes label is **held out of
training** and used only to validate the clusters afterwards.

## Files

| File | What it is |
|---|---|
| `patient_clustering.ipynb` | Full pipeline: exploration, cleaning, feature selection, scaling, K-Means + Hierarchical, evaluation, profiling, interpretation, visualization. Run top to bottom. |
| `app.py` | Streamlit app with 4 pages: Overview, Cluster Explorer, Patient Risk Assessment, Model Evaluation. |
| `data/` | The Kaggle CSVs. |
| `models/` | Artifacts written by the notebook's last section — the app loads these. |
| `outputs/` | Profile / evaluation / insight tables as CSV. |

## Setup

```bash
pip install -r requirements.txt   # app runtime
pip install jupyter               # only needed to run the notebook
```

`requirements.txt` pins exact versions because `models/*.pkl` are unpickled at startup —
scikit-learn warns or breaks when the loading version differs from the training version.

### Get the data

```bash
# Option A: Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset -p data --unzip

# Option B: direct endpoint
mkdir -p data
curl -L -o data/kaggle_diabetes.zip \
  "https://www.kaggle.com/api/v1/datasets/download/alexteboul/diabetes-health-indicators-dataset"
unzip -o data/kaggle_diabetes.zip -d data
```

## Run

```bash
# 1. train — run every cell; the last section writes models/
jupyter notebook patient_clustering.ipynb

# 2. serve
streamlit run app.py
```

The app needs `models/` to exist. Those artifacts are committed to this repo, so the app runs
straight after a clone — rerunning the notebook just regenerates them.

`data/` is gitignored (29 MB of raw CSVs the app never reads), so download the dataset before
running the notebook.

## Deploy

The app is deployed on Streamlit Community Cloud from this repo.

To deploy your own copy:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** -> pick this repo, branch `main`, main file `app.py`.
3. Deploy. Streamlit installs `requirements.txt` and serves the app.

No secrets or environment variables are needed — the model artifacts are in the repo.

## Pipeline

```
Patient dataset (253,680 respondents)
  -> Data exploration
  -> Missing value handling (0 nulls; 23,899 duplicate survey rows dropped -> 229,781)
  -> Outlier investigation (BMI winsorized at 1st/99th pct; patients kept, not deleted)
  -> Clinical feature selection (13 features; target + socio-economic dropped)
  -> Encoding + scaling (StandardScaler; ordinals kept ordered, no one-hot)
  -> K-Means  |  Hierarchical (Ward, 5,000-patient sample)
  -> Evaluation: Silhouette / Davies-Bouldin / Calinski-Harabasz
  -> Best: K-Means, k = 3
  -> Cluster profiling -> clinical interpretation -> visualization -> healthcare insights
```

## Results

Three tiers, with diabetes+prediabetes prevalence on the **held-out** label:

| Tier | Patients | Share | Diabetes+Pre | Mean BMI | HighBP | DiffWalk | GenHlth |
|---|---|---|---|---|---|---|---|
| Low Risk | 109,213 | 47.5% | 5.7% | 27.0 | 2.9% | 3.2% | 2.11 |
| Moderate Risk | 81,261 | 35.4% | 23.8% | 29.5 | 91.9% | 10.0% | 2.60 |
| High Risk | 39,307 | 17.1% | 35.9% | 31.1 | 67.6% | 78.8% | 3.97 |

Prevalence rises monotonically across tiers on a label the model never saw.

**The non-obvious finding:** hypertension does *not* separate High Risk from Moderate Risk —
the Moderate cluster has more of it (91.9% vs 67.6%). The separating axes are functional
status (`DiffWalk`, `PhysHlth`, `GenHlth`) and realised cardiovascular events. A risk score
built on biometrics alone would merge these two groups.

## Limitations

- BRFSS 2015 is self-reported; conditions are under-reported and BMI comes from self-reported
  height and weight.
- Cross-sectional, so nothing here is causal — clusters describe co-occurring risk, not progression.
- Silhouette is ~0.13, normal for mixed binary/ordinal health data where the true structure is
  a risk continuum rather than separated blobs. The output is useful risk *strata*, not natural kinds.
- `Age` is bucketed 1-13, not exact years.
- Cluster ids permute if the random seed changes; key downstream logic on the risk tier.
- Not a diagnostic tool. Clinical decisions require laboratory confirmation.
