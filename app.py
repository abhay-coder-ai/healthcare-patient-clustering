"""
Streamlit app: Healthcare Patient Clustering by Clinical Indicators.

Loads the K-Means model trained in patient_clustering.ipynb and lets a user
explore the patient segments or assign a new patient to a risk tier.

Run with:  streamlit run app.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Patient Clustering", page_icon="+", layout="wide")

AGE_BUCKETS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
    13: "80+",
}
GEN_HLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
TIER_COLOR = {"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"}

FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "BMI": "BMI",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease / heart attack",
    "GenHlth": "General health (1=excellent, 5=poor)",
    "MentHlth": "Poor mental health days (of 30)",
    "PhysHlth": "Poor physical health days (of 30)",
    "DiffWalk": "Difficulty walking / climbing stairs",
    "Age": "Age bucket (1-13)",
    "Smoker": "Smoked 100+ cigarettes in life",
    "PhysActivity": "Physical activity in past 30 days",
}


@st.cache_resource
def load_model():
    km = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    meta["cluster_risk"] = {int(k): v for k, v in meta["cluster_risk"].items()}
    return km, scaler, pca, meta


@st.cache_data
def load_tables():
    profile = pd.read_csv("models/cluster_profiles.csv", index_col=0)
    profile_z = pd.read_csv("models/cluster_profiles_z.csv", index_col=0)
    validation = pd.read_csv("models/cluster_validation.csv", index_col=0)
    scores = pd.read_csv("models/evaluation_scores.csv", index_col=0)
    algos = pd.read_csv("models/algorithm_comparison.csv")
    sample = pd.read_csv("models/clustered_sample.csv")
    return profile, profile_z, validation, scores, algos, sample


try:
    km, scaler, pca, meta = load_model()
    profile, profile_z, validation, scores, algos, sample = load_tables()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run every cell of `patient_clustering.ipynb` first — "
        "its last section writes the files this app loads into `models/`."
    )
    st.stop()

FEATURES = meta["clinical_features"]
RISK = meta["cluster_risk"]

st.sidebar.title("Patient Clustering")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Cluster Explorer", "Patient Risk Assessment", "Model Evaluation"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"K-Means, k={meta['best_k']}  \n"
    f"{meta['n_patients']:,} patients  \n"
    f"{len(FEATURES)} clinical features  \n"
    "CDC BRFSS 2015 (Kaggle)"
)


# ----------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Healthcare Patient Clustering by Clinical Indicators")
    st.markdown(
        "Unsupervised segmentation of **CDC BRFSS 2015** respondents into clinical risk "
        "tiers. The diabetes label was held out of training and used only to validate "
        "the clusters afterwards."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{meta['n_patients']:,}")
    c2.metric("Clinical features", len(FEATURES))
    c3.metric("Clusters", meta["best_k"])
    c4.metric("Silhouette", f"{meta['metrics']['silhouette']:.3f}")

    st.markdown("---")
    st.subheader("Patient segments")

    cols = st.columns(meta["best_k"])
    for col, c in zip(cols, sorted(RISK)):
        tier = RISK[c]
        n = int(profile.loc[c, "n_patients"])
        share = n / meta["n_patients"] * 100
        dpre = validation.loc[c, "Diabetes+Pre %"]
        col.markdown(
            f"<div style='border-left:6px solid {TIER_COLOR[tier]};padding:0.6rem 1rem;"
            f"background:rgba(128,128,128,0.08);border-radius:4px'>"
            f"<h4 style='margin:0'>Cluster {c}</h4>"
            f"<p style='margin:0;color:{TIER_COLOR[tier]};font-weight:700'>{tier}</p>"
            f"<p style='margin:0.4rem 0 0 0'>{n:,} patients ({share:.1f}%)<br>"
            f"Diabetes + prediabetes: <b>{dpre:.1f}%</b><br>"
            f"Mean BMI: <b>{profile.loc[c, 'BMI']:.1f}</b><br>"
            f"High BP: <b>{profile.loc[c, 'HighBP'] * 100:.0f}%</b></p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Cluster sizes")
        sizes = profile["n_patients"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            sizes,
            labels=[f"C{c}\n{RISK[c]}" for c in sizes.index],
            autopct="%1.1f%%",
            colors=[TIER_COLOR[RISK[c]] for c in sizes.index],
            startangle=90,
        )
        st.pyplot(fig)

    with right:
        st.subheader("Diabetes prevalence by cluster")
        st.caption("Label never used in training — rising prevalence validates the tiers.")
        vcols = ["No Diabetes %", "Prediabetes %", "Diabetes %"]
        fig, ax = plt.subplots(figsize=(5, 4))
        validation[vcols].plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("% of cluster")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Pipeline")
    st.code(
        """1. Load dataset (253,680 BRFSS respondents)
2. Understand & explore data
3. Preprocessing (0 nulls; 23,899 duplicates dropped; BMI capped at 1st/99th pct;
   random sample of 10,000 so all three algorithms run on identical data)
4. Select clinical features (12; target + socio-economic + access dropped)
5. Feature scaling (StandardScaler; ordinals kept ordered)
6. Three clustering algorithms
     K-Means      -> elbow method      -> k = 3
     Hierarchical -> dendrogram        -> k = 3
     DBSCAN       -> k-distance graph  -> eps = 2.5
7. Evaluate all three (Silhouette Score)
8. Compare models
9. Best: K-Means k = 3
10. Visualize clusters
11. Profile & interpret patient clusters
12. Final conclusion""",
        language="text",
    )

    with st.expander("Features used for clustering"):
        st.dataframe(
            pd.DataFrame(
                {"Feature": FEATURES, "Meaning": [FEATURE_LABELS[f] for f in FEATURES]}
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Features deliberately excluded"):
        st.markdown(
            "- `Diabetes_012` — target, held out for validation\n"
            "- `Income`, `Education` — socio-economic, not clinical\n"
            "- `AnyHealthcare`, `NoDocbcCost` — healthcare access, not patient health\n"
            "- `Fruits`, `Veggies` — self-reported diet, near-zero correlation with outcome\n"
            "- `CholCheck` — ~96% of patients = 1, almost no variance\n"
            "- `Sex` — demographic; would split clusters by gender instead of by risk\n"
            "- `HvyAlcoholConsump` — only ~6% = 1; K-Means carved it into its own cluster, "
            "hiding the risk structure"
        )


# --------------------------------------------------------- Cluster Explorer
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")

    st.subheader("Cluster profiles (z-score vs. population mean)")
    st.caption("Red = worse than the population average on that indicator.")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    sns.heatmap(profile_z, cmap="RdYlGn_r", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_ylabel("Cluster")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Clusters in PCA space")
    cL, cR = st.columns([2, 1])
    with cR:
        show = st.multiselect(
            "Clusters to plot", sorted(RISK), default=sorted(RISK),
            format_func=lambda c: f"C{c} - {RISK[c]}",
        )
        n_pts = st.slider("Points plotted", 1000, len(sample), 8000, step=1000)
    with cL:
        sub = sample[sample["Cluster"].isin(show)].head(n_pts)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for c in show:
            s = sub[sub["Cluster"] == c]
            ax.scatter(s["PC1"], s["PC2"], s=8, alpha=0.4,
                       color=TIER_COLOR[RISK[c]], label=f"C{c} - {RISK[c]}")
        cent = pca.transform(km.cluster_centers_)
        for c in show:
            ax.scatter(*cent[c], marker="X", s=280, c="black", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Compare one indicator across clusters")
    feat = st.selectbox("Indicator", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        profile[feat].plot(kind="bar", ax=ax,
                           color=[TIER_COLOR[RISK[c]] for c in profile.index])
        ax.set_title(f"Mean {feat} by cluster")
        ax.set_xlabel("Cluster")
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for c in sorted(RISK):
            sns.kdeplot(sample[sample["Cluster"] == c][feat], ax=ax, fill=True,
                        alpha=0.25, color=TIER_COLOR[RISK[c]], label=f"C{c}")
        ax.set_title(f"{feat} distribution by cluster")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Per-cluster detail")
    pick = st.selectbox("Cluster", sorted(RISK), format_func=lambda c: f"Cluster {c} - {RISK[c]}")
    z = profile_z.loc[pick]
    hi = z[z > 0.4].sort_values(ascending=False)
    lo = z[z < -0.4].sort_values()

    d1, d2, d3 = st.columns(3)
    d1.metric("Patients", f"{int(profile.loc[pick, 'n_patients']):,}")
    d2.metric("Diabetes + prediabetes", f"{validation.loc[pick, 'Diabetes+Pre %']:.1f}%")
    d3.metric("Risk tier", RISK[pick])

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Elevated vs. population**")
        if len(hi):
            for k, v in hi.items():
                st.markdown(f"- {FEATURE_LABELS[k]} — **+{v:.2f} sd**")
        else:
            st.markdown("_Nothing above +0.4 sd._")
    with e2:
        st.markdown("**Below population**")
        if len(lo):
            for k, v in lo.items():
                st.markdown(f"- {FEATURE_LABELS[k]} — **{v:.2f} sd**")
        else:
            st.markdown("_Nothing below -0.4 sd._")

    st.markdown("**Mean of every indicator (original units)**")
    st.dataframe(profile.loc[[pick]].T, width="stretch")


# ------------------------------------------------- Patient Risk Assessment
elif page == "Patient Risk Assessment":
    st.title("Patient Risk Assessment")
    st.markdown(
        "Enter a patient's clinical indicators to assign them to a segment. "
        "This is a **clustering** model — it reports which existing patient group this "
        "person resembles, not an individual diagnosis or probability."
    )

    with st.form("patient"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Vitals & diagnoses**")
            bmi = st.slider("BMI", 12.0, 60.0, 28.0, 0.5)
            high_bp = st.radio("High blood pressure", [0, 1],
                               format_func=lambda v: "Yes" if v else "No", horizontal=True)
            high_chol = st.radio("High cholesterol", [0, 1],
                                 format_func=lambda v: "Yes" if v else "No", horizontal=True)
            stroke = st.radio("History of stroke", [0, 1],
                              format_func=lambda v: "Yes" if v else "No", horizontal=True)
            chd = st.radio("Heart disease or heart attack", [0, 1],
                           format_func=lambda v: "Yes" if v else "No", horizontal=True)

        with c2:
            st.markdown("**Health status**")
            gen = st.select_slider("General health", options=[1, 2, 3, 4, 5], value=3,
                                   format_func=lambda v: f"{v} - {GEN_HLTH[v]}")
            phys_h = st.slider("Poor physical health days (last 30)", 0, 30, 0)
            ment_h = st.slider("Poor mental health days (last 30)", 0, 30, 0)
            diff_walk = st.radio("Difficulty walking or climbing stairs", [0, 1],
                                 format_func=lambda v: "Yes" if v else "No", horizontal=True)
            age = st.select_slider("Age", options=list(range(1, 14)), value=7,
                                   format_func=lambda v: f"{v} - {AGE_BUCKETS[v]}")

        with c3:
            st.markdown("**Behaviour**")
            smoker = st.radio("Smoked 100+ cigarettes in life", [0, 1],
                              format_func=lambda v: "Yes" if v else "No", horizontal=True)
            phys_act = st.radio("Physical activity in past 30 days", [0, 1],
                                format_func=lambda v: "Yes" if v else "No",
                                index=1, horizontal=True)

        submitted = st.form_submit_button("Assign to cluster", type="primary")

    if submitted:
        lo_cap, hi_cap = meta["bmi_cap"]
        values = {
            "HighBP": high_bp, "HighChol": high_chol,
            "BMI": float(np.clip(bmi, lo_cap, hi_cap)),
            "Stroke": stroke, "HeartDiseaseorAttack": chd, "GenHlth": gen,
            "MentHlth": ment_h, "PhysHlth": phys_h, "DiffWalk": diff_walk,
            "Age": age, "Smoker": smoker, "PhysActivity": phys_act,
        }
        patient = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        scaled = scaler.transform(patient)
        cluster = int(km.predict(scaled)[0])
        tier = RISK[cluster]

        st.markdown("---")
        st.markdown(
            f"<div style='border-left:8px solid {TIER_COLOR[tier]};padding:1rem 1.5rem;"
            f"background:rgba(128,128,128,0.08);border-radius:4px'>"
            f"<h2 style='margin:0'>Cluster {cluster} — "
            f"<span style='color:{TIER_COLOR[tier]}'>{tier}</span></h2>"
            f"<p style='margin:0.3rem 0 0 0'>This patient's clinical profile matches a group of "
            f"<b>{int(profile.loc[cluster, 'n_patients']):,}</b> patients "
            f"({profile.loc[cluster, 'n_patients'] / meta['n_patients'] * 100:.1f}% of the cohort).</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Diabetes + prediabetes in this group",
                  f"{validation.loc[cluster, 'Diabetes+Pre %']:.1f}%")
        m2.metric("Group mean BMI", f"{profile.loc[cluster, 'BMI']:.1f}",
                  f"{values['BMI'] - profile.loc[cluster, 'BMI']:+.1f} vs. patient")
        m3.metric("High BP in this group", f"{profile.loc[cluster, 'HighBP'] * 100:.0f}%")

        # Distance to each centroid - how clear-cut is the assignment?
        dists = np.linalg.norm(km.cluster_centers_ - scaled, axis=1)
        st.markdown("**Distance to each cluster centroid** (smaller = closer match)")
        dist_df = pd.DataFrame({
            "Cluster": [f"C{c} - {RISK[c]}" for c in range(len(dists))],
            "Distance": dists.round(3),
        }).sort_values("Distance")
        st.dataframe(dist_df, hide_index=True, width="stretch")
        if dists.min() / np.partition(dists, 1)[1] > 0.85:
            st.info(
                "This patient sits almost equally close to two clusters — a borderline case. "
                "Treat the tier as indicative and review the individual indicators below."
            )

        st.markdown("---")
        st.subheader("Patient vs. their cluster vs. the population")
        comp = pd.DataFrame({
            "Patient": [values[f] for f in FEATURES],
            "Cluster mean": [profile.loc[cluster, f] for f in FEATURES],
            "Population mean": [meta["population_mean"][f] for f in FEATURES],
        }, index=[FEATURE_LABELS[f] for f in FEATURES]).round(2)
        st.dataframe(comp, width="stretch")

        st.subheader("Where this patient deviates from the population")
        pz = pd.Series(
            {f: (values[f] - meta["population_mean"][f]) / meta["population_std"][f]
             for f in FEATURES}
        ).sort_values()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(
            [FEATURE_LABELS[f] for f in pz.index], pz.values,
            color=["#e74c3c" if v > 0 else "#2ecc71" for v in pz.values],
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Standard deviations from population mean")
        st.pyplot(fig)

        st.caption(
            "Not a diagnostic tool. Built on self-reported BRFSS 2015 survey data; "
            "clinical decisions require laboratory confirmation."
        )


# --------------------------------------------------------- Model Evaluation
else:
    st.title("Model Evaluation")

    algo_meta = meta["algorithms"]

    st.subheader("Final model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "K-Means")
    c2.metric("Clusters (k)", meta["best_k"])
    c3.metric("Silhouette", f"{meta['metrics']['silhouette']:.4f}", help="Higher is better")

    st.markdown("---")
    st.subheader("All three algorithms, same 10,000 patients")
    st.caption(
        "Silhouette measures how close each patient is to its own cluster versus the "
        "nearest other cluster. Range -1 to 1, higher is better."
    )
    st.dataframe(algos, hide_index=True, width="stretch")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    bar_colors = ["#3498db", "#9b59b6", "#e67e22"]
    axes[0].bar(algos["Algorithm"], algos["Silhouette"], color=bar_colors)
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette by algorithm (higher is better)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(algos["Silhouette"]):
        axes[0].text(i, v + 0.004, f"{v:.3f}", ha="center")

    axes[1].bar(algos["Algorithm"], algos["Largest cluster %"], color=bar_colors)
    axes[1].axhline(80, color="red", ls="--", label="80% - cluster is the whole cohort")
    axes[1].set_ylabel("% of patients in the biggest cluster")
    axes[1].set_title("Cluster balance (lower is better)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("How each algorithm picked its parameter")
    p1, p2, p3 = st.columns(3)
    p1.markdown(
        f"**K-Means — elbow method**\n\n"
        f"Inertia bends at k = {algo_meta['kmeans']['k']}.\n\n"
        f"Silhouette: `{algo_meta['kmeans']['silhouette']:.4f}`"
    )
    p2.markdown(
        f"**Hierarchical — dendrogram**\n\n"
        f"Longest vertical gap splits the tree into {algo_meta['hierarchical']['k']} branches.\n\n"
        f"Silhouette: `{algo_meta['hierarchical']['silhouette']:.4f}`"
    )
    p3.markdown(
        f"**DBSCAN — k-distance graph**\n\n"
        f"Knee at eps = {algo_meta['dbscan']['eps']}, min_samples = "
        f"{algo_meta['dbscan']['min_samples']} "
        f"({algo_meta['dbscan']['n_clusters']} clusters, "
        f"{algo_meta['dbscan']['noise']} noise points).\n\n"
        f"Silhouette: `{algo_meta['dbscan']['silhouette']:.4f}`"
    )

    st.markdown("---")
    st.subheader("Silhouette across k")
    st.caption("DBSCAN is absent because it is not given a number of clusters.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scores.index, scores["kmeans_silhouette"], "o-", label="K-Means")
    ax.plot(scores.index, scores["hier_silhouette"], "s--", label="Hierarchical")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.dataframe(scores, width="stretch")

    st.markdown("---")
    st.subheader("Why K-Means with k = 3")
    st.markdown(
        "Silhouette alone would pick Hierarchical (0.249 vs 0.142). The score is misleading "
        "here: that solution puts **72% of patients in one cluster**, which gives a care team "
        "nothing to act on. DBSCAN (0.226) has the same problem — 74% in one cluster — and "
        "additionally labels 566 patients as noise, because it looks for dense blobs separated "
        "by empty space and health risk is a continuum with no empty space in it.\n\n"
        "**Why K-Means wins anyway.**\n"
        "- **Balance.** 17% / 34% / 49% — three segments a care program can actually staff.\n"
        "- **New-patient assignment.** K-Means leaves behind centroids, so an unseen patient is "
        "assigned with one distance computation. Hierarchical and DBSCAN produce labels but no "
        "model, so serving them would mean refitting on every new patient. This app needs it.\n"
        "- **All three scores are low (0.14-0.25),** which is normal for mixed binary/ordinal "
        "health data. None of them found well-separated natural groups, so balance and "
        "usability decide.\n\n"
        "**Why k = 3.** k = 2 scores higher (0.275) but collapses into a healthy/unhealthy "
        "binary with no middle tier — exactly the group where intervention still changes the "
        "outcome. k = 3 is the elbow and the smallest k that produces that tier."
    )

    st.subheader("Validation against the held-out label")
    st.caption("`Diabetes_012` was excluded from training entirely.")
    st.dataframe(validation, width="stretch")

    st.markdown("---")
    st.subheader("Limitations")
    st.markdown(
        "- BRFSS 2015 is **self-reported**: conditions are under-reported and BMI comes from "
        "self-reported height and weight.\n"
        "- Cross-sectional data, so nothing here is causal — clusters describe co-occurring "
        "risk, not disease progression.\n"
        "- Built on a random sample of 10,000 of the 229,781 de-duplicated respondents, "
        "because Hierarchical and DBSCAN need an O(n^2) distance matrix.\n"
        "- Silhouette values are modest, which is normal for mixed binary/ordinal health data "
        "where the true structure is a risk continuum rather than well-separated blobs. "
        "The output is useful **risk strata**, not natural kinds.\n"
        "- `Age` is bucketed (1-13), not exact years.\n"
        "- Cluster ids permute between training runs; downstream logic should key on the "
        "risk tier, not the id."
    )
