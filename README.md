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
| `patient_clustering.ipynb` | Full pipeline: exploration, cleaning, feature selection, scaling, K-Means + Hierarchical + DBSCAN, Silhouette evaluation, model comparison, profiling, interpretation, visualization. Run top to bottom. |
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
 1. Load dataset (253,680 respondents)
 2. Understand & explore data
 3. Data preprocessing
      0 nulls; 23,899 duplicate survey rows dropped -> 229,781
      BMI capped at the 1st/99th percentile (patients kept, not deleted)
      random sample of 10,000 so all three algorithms run on identical data
 4. Select clinical features (12; target + socio-economic + access dropped)
 5. Feature scaling (StandardScaler; ordinals kept ordered, no one-hot)
 6. Apply three clustering algorithms
      K-Means      -> elbow method      -> k = 3
      Hierarchical -> dendrogram        -> k = 3
      DBSCAN       -> k-distance graph  -> eps = 2.5, min_samples = 24
 7. Evaluate all three (Silhouette Score)
 8. Compare models
 9. Select best: K-Means, k = 3
10. Visualize clusters (PCA)
11. Profile & interpret patient clusters
12. Final conclusion
```

**Why a 10,000-patient sample.** Hierarchical clustering and DBSCAN both need the full
pairwise distance matrix — O(n^2), roughly 250 GB at 229,781 patients. Comparing the three
algorithms fairly means running them on the same data, so the whole project is built on the
sample.

## Model comparison

| Algorithm | Parameter from | Silhouette | Biggest cluster | Noise |
|---|---|---|---|---|
| **K-Means, k=3** | elbow method | 0.142 | 49.0% | 0 |
| Hierarchical, k=3 | dendrogram | 0.249 | 72.1% | 0 |
| DBSCAN, eps=2.5 | k-distance graph | 0.226 | 73.9% | 566 |

Silhouette alone would pick Hierarchical. The score is misleading here: it rewards a solution
that puts 72% of patients in one cluster, which gives a care team nothing to act on. DBSCAN
has the same problem and additionally labels 566 patients as noise — it looks for dense blobs
separated by empty space, and health risk is a continuum with no empty space in it.

K-Means was chosen because it produces three balanced, usable segments, and because it is the
only one of the three that leaves behind a model (centroids) that can assign a new patient.

## Results

Three tiers, with diabetes+prediabetes prevalence on the **held-out** label:

| Tier | Patients | Share | Diabetes+Pre | Mean BMI | HighBP | DiffWalk | GenHlth |
|---|---|---|---|---|---|---|---|
| Low Risk | 4,902 | 49.0% | 6.1% | 27.2 | 1.1% | 3.6% | 2.13 |
| Moderate Risk | 3,377 | 33.8% | 23.1% | 29.3 | 96.7% | 9.2% | 2.59 |
| High Risk | 1,721 | 17.2% | 36.0% | 31.1 | 66.2% | 78.3% | 3.91 |

Prevalence rises monotonically across tiers on a label the model never saw.

**The non-obvious finding:** hypertension does *not* separate High Risk from Moderate Risk —
the Moderate cluster has more of it (96.7% vs 66.2%). The separating axes are functional
status (`DiffWalk`, `PhysHlth`, `GenHlth`) and realised cardiovascular events. A risk score
built on biometrics alone would merge these two groups.

## Limitations

- BRFSS 2015 is self-reported; conditions are under-reported and BMI comes from self-reported
  height and weight.
- Cross-sectional, so nothing here is causal — clusters describe co-occurring risk, not progression.
- Built on a random sample of 10,000 of the 229,781 de-duplicated respondents (see above).
- Silhouette is ~0.14, normal for mixed binary/ordinal health data where the true structure is
  a risk continuum rather than separated blobs. The output is useful risk *strata*, not natural kinds.
- `Age` is bucketed 1-13, not exact years.
- Cluster ids permute if the random seed changes; key downstream logic on the risk tier.
- Not a diagnostic tool. Clinical decisions require laboratory confirmation.
