# Healthcare Patient Clustering by Clinical Indicators

Unsupervised segmentation of CDC BRFSS 2015 survey respondents into clinical risk tiers, with
a Streamlit app for exploring the segments and assigning a new patient to one.

**Live app:** https://abhay-coder-ai-healthcare-patient-clustering-app-moztbw.streamlit.app/

**Repository:** https://github.com/abhay-coder-ai/healthcare-patient-clustering

---

## Problem

Clinics hold a lot of survey and routine-vitals data but no lab results for most patients.
The question this project answers: **can patients be sorted into useful risk groups from that
data alone, without a blood test and without being told who already has diabetes?**

The diabetes column is removed before any clustering and brought back only at the end, as a
check on whether the groups mean anything clinically.

## Dataset

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(Kaggle, `alexteboul`) — the CDC Behavioral Risk Factor Surveillance System 2015 survey.

| | |
|---|---|
| File used | `diabetes_012_health_indicators_BRFSS2015.csv` |
| Rows | 253,680 respondents |
| Columns | 22 (21 indicators + the diabetes label) |
| Target | `Diabetes_012` — 0 no diabetes, 1 prediabetes, 2 diabetes |
| Target use | **held out of training**, used only to validate the clusters |

## Pipeline

```
 1. Load dataset                      253,680 respondents
 2. Understand and explore data       distributions, prevalence, correlations
 3. Data preprocessing                0 nulls
                                      23,899 duplicate rows dropped -> 229,781
                                      BMI capped at 1st/99th percentile (18 to 50)
                                      random sample of 10,000 patients
 4. Select clinical features          12 features kept
 5. Feature scaling                   StandardScaler
 6. Apply three clustering algorithms
      K-Means        elbow method       -> k = 3
      Hierarchical   dendrogram         -> k = 3
      DBSCAN         k-distance graph   -> eps = 2.5, min_samples = 24
 7. Evaluate all three                 Silhouette Score
 8. Compare models                     score and cluster balance
 9. Select best clustering model        K-Means, k = 3
10. Visualize clusters                  PCA to 2 components
11. Profile and interpret clusters      means, z-scores, risk tiers
12. Final conclusion
```

### Why a 10,000-patient sample

Hierarchical clustering and DBSCAN both need the distance between every pair of points — an
O(n²) matrix, roughly 250 GB at 229,781 patients. Comparing the three algorithms fairly means
running them on the same rows, so the whole project is built on the sample.

### Why BMI is capped and not trimmed

A BMI of 60–98 is clinically real (morbid obesity) and is exactly the high-risk group a cluster
is wanted for, so those patients are kept. K-Means is distance-based, though, so a few rows at
BMI 98 would stretch that axis and pull a centroid towards them. Capping at the 1st and 99th
percentile keeps the patients and removes the leverage.

## Features

**Kept (12)** — a diagnosed condition, a biometric, a functional-status measure, or a
behavioural risk factor:

| Feature | Meaning |
|---|---|
| `HighBP` | diagnosed hypertension |
| `HighChol` | diagnosed high cholesterol |
| `BMI` | body mass index |
| `Stroke` | history of stroke |
| `HeartDiseaseorAttack` | coronary heart disease or myocardial infarction |
| `GenHlth` | self-rated general health, 1 excellent to 5 poor |
| `MentHlth` | days of poor mental health in last 30 |
| `PhysHlth` | days of poor physical health in last 30 |
| `DiffWalk` | difficulty walking or climbing stairs |
| `Age` | age bucket, 1 (18-24) to 13 (80+) |
| `Smoker` | smoked 100+ cigarettes in life |
| `PhysActivity` | physical activity in past 30 days |

**Dropped (9)**

| Dropped | Reason |
|---|---|
| `Diabetes_012` | target, held out for validation |
| `Income`, `Education` | socio-economic, not a clinical measurement |
| `AnyHealthcare`, `NoDocbcCost` | healthcare *access*, not patient health |
| `Fruits`, `Veggies` | self-reported diet, near-zero correlation with the outcome |
| `CholCheck` | ~96% of patients are 1, almost no variance to cluster on |
| `Sex` | demographic; would split clusters by gender instead of by risk |
| `HvyAlcoholConsump` | only ~6% are 1, and K-Means carved those patients into a cluster of their own, hiding the risk structure |

## Model comparison

All three algorithms run on the same 10,000 patients and the same 12 scaled features, so the
scores are directly comparable.

| Algorithm | Parameter chosen by | Silhouette | Clusters | Biggest cluster | Noise |
|---|---|---|---|---|---|
| **K-Means, k=3** | elbow method | 0.1418 | 3 | 49.0% | 0 |
| Hierarchical (Ward), k=3 | dendrogram | 0.2486 | 3 | 72.1% | 0 |
| DBSCAN, eps=2.5 | k-distance graph | 0.2264 | 5 | 73.9% | 566 |

### Selected model: K-Means with k = 3

Silhouette alone would pick Hierarchical. The score is misleading here:

- **Hierarchical** puts **72% of patients in one cluster**. A high score for a split that gives
  a care team one undifferentiated mass is not a useful split.
- **DBSCAN** has the same problem (74% in one cluster) and additionally labels **566 patients
  as noise**. DBSCAN looks for dense blobs separated by empty space; health risk is a continuum,
  so there is no empty space for it to find.
- **K-Means** gives three balanced segments — 49% / 34% / 17% — that a care programme can
  actually be staffed around.
- Only K-Means leaves behind a **model**. Its centroids assign an unseen patient with one
  distance computation. The other two produce labels but nothing reusable, so serving them
  would mean refitting on every new patient. The app requires assignment.
- All three scores are low (0.14–0.25), which is normal for mixed binary and ordinal health
  data. None of them found well-separated natural groups, so balance and usability decide.

**Why k = 3.** k = 2 scores higher (0.2748) but collapses into healthy / unhealthy with no
middle tier — exactly the group where intervention still changes the outcome. k = 3 is the
elbow and the smallest k that produces that tier.

Silhouette across k:

| k | K-Means | Hierarchical |
|---|---|---|
| 2 | 0.2748 | 0.2495 |
| **3** | **0.1418** | 0.2486 |
| 4 | 0.1539 | 0.2557 |
| 5 | 0.1304 | 0.1236 |
| 6 | 0.1310 | 0.1300 |
| 7 | 0.1462 | 0.1470 |
| 8 | 0.1404 | 0.1272 |

## Results

Three risk tiers. Diabetes + prediabetes prevalence is measured on the **held-out** label.

| Tier | Cluster | Patients | Share | Diabetes+Pre | BMI | HighBP | HighChol | CVD | DiffWalk | PhysHlth | GenHlth | Age |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low Risk | 2 | 4,902 | 49.0% | **6.1%** | 27.2 | 1.1% | 28.9% | 1.8% | 3.6% | 1.6 | 2.13 | 6.9 |
| Moderate Risk | 1 | 3,377 | 33.8% | **23.1%** | 29.3 | 96.7% | 57.1% | 11.4% | 9.2% | 1.8 | 2.59 | 9.4 |
| High Risk | 0 | 1,721 | 17.2% | **36.0%** | 31.1 | 66.2% | 63.6% | 30.9% | 78.3% | 18.8 | 3.91 | 9.2 |

Prevalence rises monotonically across the tiers on a label no algorithm ever saw. The segments
carry real clinical signal rather than being an artefact of the arithmetic.

### Reading the tiers

**Low Risk — healthy younger adults.** Almost no hypertension, lowest cholesterol, mean BMI 27,
hardly any cardiovascular history, youngest age buckets, highest physical activity. Below the
population average on every risk feature.

**Moderate Risk — metabolically burdened, functionally intact.** Hypertension is near-universal
at 96.7% with 57% high cholesterol and mean BMI 29.3, but function is preserved: only 9% report
difficulty walking and poor-health days stay below the population mean.

**High Risk — multi-morbid, functionally limited.** 31% cardiovascular disease, 13% stroke
history, mean BMI 31, general health 3.91 ("fair"), nearly 19 poor physical-health days a
month, and 78% report difficulty walking. Disease here has already cost function.

### The non-obvious finding

Hypertension does **not** separate High Risk from Moderate Risk — the Moderate tier has *more*
of it (96.7% vs 66.2%). What separates them is **functional status** (`DiffWalk`, `PhysHlth`,
`GenHlth`) and realised cardiovascular events. A risk score built on biometrics alone would
merge these two groups and miss that one of them can still be prevented and the other has to
be managed.

### What it is useful for

Every feature comes from a survey or routine vitals — no HbA1c, no fasting glucose. A clinic
can tier its whole patient panel from intake data and send expensive confirmatory testing to
the High Risk tier first. The Moderate tier is the intervention target: it carries the full
metabolic load but has not yet lost function, so lifestyle and drug therapy can still change
its trajectory.

## Project structure

```
DMDW_Project/
├── patient_clustering.ipynb    full pipeline, 12 steps, run top to bottom
├── app.py                      Streamlit app, 4 pages
├── requirements.txt            app runtime dependencies
├── data/                       Kaggle CSVs (gitignored, 56 MB)
├── models/                     artifacts the app loads
│   ├── kmeans_model.pkl        trained K-Means (k=3)
│   ├── scaler.pkl              fitted StandardScaler
│   ├── pca.pkl                 fitted PCA (2 components)
│   ├── metadata.json           features, risk tiers, scores, population stats
│   ├── cluster_profiles.csv    mean of every feature per cluster
│   ├── cluster_profiles_z.csv  same table as z-scores
│   ├── cluster_validation.csv  diabetes prevalence per cluster
│   ├── evaluation_scores.csv   silhouette across k for both k-based models
│   ├── algorithm_comparison.csv  the three-way comparison
│   └── clustered_sample.csv    10,000 patients with cluster label and PCA coords
└── outputs/                    profile, evaluation and summary tables as CSV
```

## The app

Four pages:

| Page | What it does |
|---|---|
| **Overview** | Cohort size, the three segments as cards, cluster sizes, diabetes prevalence, the pipeline, features used and excluded |
| **Cluster Explorer** | z-score profile heatmap, clusters in PCA space with centroids, one indicator compared across clusters, per-cluster detail |
| **Patient Risk Assessment** | Enter a patient's 12 indicators, get their cluster and risk tier, distance to each centroid, and how they compare with their cluster and the population |
| **Model Evaluation** | The three-algorithm comparison, how each parameter was chosen, silhouette across k, the reasoning behind the choice, validation against the held-out label, limitations |

## Setup

```bash
pip install -r requirements.txt   # app runtime
pip install jupyter               # only needed to run the notebook
```

`scikit-learn` is pinned exactly, because `models/*.pkl` are unpickled at startup and
scikit-learn warns or breaks when the loading version differs from the training version. The
other versions are floors chosen so the Streamlit Cloud image (Python 3.14) can install wheels
instead of compiling from source.

### Get the data

`data/` is gitignored (56 MB of raw CSVs the app never reads), so download it before running
the notebook. The app itself does not need it.

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
# train — run every cell; the last section writes models/
jupyter notebook patient_clustering.ipynb

# serve
streamlit run app.py
```

`models/` is committed, so the app runs straight after a clone. Rerunning the notebook just
regenerates those files.

Everything is seeded (`random_state=42`), so a rerun reproduces the numbers in this README
exactly.

## Deploy

Deployed on Streamlit Community Cloud from this repository. Each push to `main` redeploys
automatically; if a deploy is sleeping, opening the URL wakes it.

To deploy your own copy:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → this repo, branch `main`, main file `app.py`.
3. Deploy. Streamlit installs `requirements.txt` and serves it.

No secrets or environment variables are needed — the model artifacts are in the repo.

## Tech stack

pandas · numpy · matplotlib · scikit-learn (KMeans, AgglomerativeClustering, DBSCAN,
StandardScaler, PCA, silhouette_score, NearestNeighbors) · scipy (Ward linkage, dendrogram) ·
joblib · Streamlit

## Limitations

- BRFSS 2015 is **self-reported**. Conditions are under-reported and BMI comes from
  self-reported height and weight.
- Built on a random sample of 10,000 of the 229,781 de-duplicated respondents, because
  Hierarchical and DBSCAN need the full O(n²) distance matrix.
- The data is **cross-sectional**, so nothing here is causal. The clusters describe
  co-occurring risk, not disease progression.
- Silhouette values are modest (0.14–0.25), which is expected for mixed binary and ordinal
  health data where the true structure is a risk continuum rather than separated blobs. The
  output is useful risk **strata**, not natural kinds.
- `Age` is a bucket from 1 to 13, not exact years.
- Cluster **ids** permute if the random seed changes. Downstream logic should key on the risk
  tier, never on the id.
- **Not a diagnostic tool.** It reports which existing patient group a person resembles, not an
  individual diagnosis or probability. Clinical decisions require laboratory confirmation.
