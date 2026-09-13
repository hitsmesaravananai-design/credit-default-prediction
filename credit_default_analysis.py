"""
Group Project Part 1 - Real-World AI-Driven Analytics Solution Using Python
Problem: Predicting credit card default risk (UCI "Default of Credit Card Clients")

Pipeline
--------
1. Load raw data
2. Explore (summary stats, class balance, missing values, distributions, correlations)
3. Clean and preprocess (fix undocumented category codes, engineer features,
   encode, scale, stratified train/test split)
4. Build initial models (Logistic Regression baseline, Random Forest)
5. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
   5-fold cross-validation) and check fairness across sex / age groups
6. Save figures, metrics, and cleaned data for the report

Run:  python credit_default_analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", palette="deep")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "default of credit card clients.xls"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
results: dict = {}


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA, header=1)
    df = df.rename(columns={"default payment next month": "DEFAULT", "PAY_0": "PAY_1"})
    df = df.drop(columns=["ID"])
    return df


# --------------------------------------------------------------------------- #
# 2. Explore
# --------------------------------------------------------------------------- #
def explore(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print("\nDtypes:\n", df.dtypes.value_counts())
    print("\nMissing values per column (total):", int(df.isna().sum().sum()))
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nTarget balance:\n", df["DEFAULT"].value_counts(normalize=True).round(4))

    desc = df[["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]].describe().round(1)
    print("\nSummary statistics (selected):\n", desc)
    desc.to_csv(OUT / "summary_stats.csv")

    # Undocumented category codes
    print("\nEDUCATION codes:", sorted(df["EDUCATION"].unique()))
    print("MARRIAGE codes:", sorted(df["MARRIAGE"].unique()))
    print("PAY_1 codes:", sorted(df["PAY_1"].unique()))

    results["n_rows"], results["n_cols"] = int(df.shape[0]), int(df.shape[1])
    results["missing_total"] = int(df.isna().sum().sum())
    results["duplicates"] = int(df.duplicated().sum())
    results["default_rate"] = float(df["DEFAULT"].mean())
    results["education_undoc"] = int(df["EDUCATION"].isin([0, 5, 6]).sum())
    results["marriage_undoc"] = int((df["MARRIAGE"] == 0).sum())
    results["limit_bal_median"] = float(df["LIMIT_BAL"].median())
    results["limit_bal_max"] = float(df["LIMIT_BAL"].max())
    results["age_min"], results["age_max"] = int(df["AGE"].min()), int(df["AGE"].max())

    # --- Figure 1: class balance
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["DEFAULT"].value_counts()
    ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 300, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=12)
    ax.set_ylabel("Clients")
    ax.set_title("Target class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_class_balance.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: distributions of key numeric features
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["LIMIT_BAL"] / 1000, bins=40, ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("Credit limit (NT$ thousands)")
    axes[0].set_title("Credit limit")
    sns.histplot(df["AGE"], bins=35, ax=axes[1], color="#55A868")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_title("Age")
    sns.histplot(df["BILL_AMT1"].clip(-50_000, 400_000) / 1000, bins=40, ax=axes[2], color="#DD8452")
    axes[2].set_xlabel("Bill amount, most recent month (NT$ thousands)")
    axes[2].set_title("Bill amount (clipped at 400k)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_distributions.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: default rate by repayment status (strongest signal)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rate = df.groupby("PAY_1")["DEFAULT"].mean()
    n = df.groupby("PAY_1").size()
    keep = n[n >= 50].index
    ax.bar(rate[keep].index.astype(str), rate[keep].values, color="#C44E52")
    ax.set_xlabel("Repayment status last month (PAY_1)\n-2/-1/0 = paid or no balance, 1..8 = months delayed")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate rises sharply with payment delay")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_default_by_pay_status.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: correlation heatmap (subset)
    cols = ["DEFAULT", "LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3",
            "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation matrix (selected features)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_correlation.png", dpi=150)
    plt.close(fig)

    results["corr_default_pay1"] = float(corr.loc["DEFAULT", "PAY_1"])
    results["corr_default_limit"] = float(corr.loc["DEFAULT", "LIMIT_BAL"])
    results["corr_bill1_bill2"] = float(corr.loc["BILL_AMT1", "BILL_AMT2"])

    # Default rate by demographic groups (for the ethics discussion)
    grp = {
        "sex": df.groupby(df["SEX"].map({1: "Male", 2: "Female"}))["DEFAULT"].mean().round(4).to_dict(),
        "education": df.groupby("EDUCATION")["DEFAULT"].mean().round(4).to_dict(),
    }
    results["default_rate_by_group"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in grp.items()}
    print("\nDefault rate by sex:", grp["sex"])


# --------------------------------------------------------------------------- #
# 3. Preprocess
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Collapse undocumented codes into "Other" categories
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # 4 = others
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                # 3 = others

    # Feature engineering
    pay_cols = [f"PAY_{i}" for i in range(1, 7)]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    df["MAX_DELAY"] = df[pay_cols].max(axis=1)
    df["N_MONTHS_DELAYED"] = (df[pay_cols] > 0).sum(axis=1)
    df["AVG_BILL"] = df[bill_cols].mean(axis=1)
    df["AVG_PAYMENT"] = df[amt_cols].mean(axis=1)
    df["UTILIZATION"] = (df["AVG_BILL"] / df["LIMIT_BAL"]).clip(-1, 5)
    df["PAY_TO_BILL_RATIO"] = (df["AVG_PAYMENT"] / df["AVG_BILL"].abs().replace(0, np.nan)).fillna(1).clip(0, 5)

    # Age bands used later for fairness checks (not a model input)
    df["AGE_BAND"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 80],
                            labels=["21-30", "31-40", "41-50", "51+"], right=True)

    df.to_csv(OUT / "cleaned_data.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 4/5. Model + evaluate
# --------------------------------------------------------------------------- #
def build_and_evaluate(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("MODEL DEVELOPMENT")
    print("=" * 70)

    target = "DEFAULT"
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    num_features = [c for c in df.columns if c not in cat_features + [target, "AGE_BAND"]]

    X = df[cat_features + num_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    results["train_size"], results["test_size"] = int(len(X_train)), int(len(X_test))
    results["n_features_raw"] = int(X.shape[1])

    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            [("prep", preproc),
             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            [("prep", preproc),
             ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                            class_weight="balanced_subsample", n_jobs=-1,
                                            random_state=RANDOM_STATE))]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics: dict[str, dict] = {}
    fitted = {}

    fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5))
    for name, pipe in models.items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "confusion": confusion_matrix(y_test, pred).tolist(),
        }
        metrics[name] = m
        print(f"\n--- {name} ---")
        print(f"5-fold CV ROC-AUC: {m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
        print(classification_report(y_test, pred, digits=4))
        print("Test ROC-AUC:", round(m["roc_auc"], 4))
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax_roc)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax_roc.set_title("ROC curves on held-out test set")
    ax_roc.legend(loc="lower right", fontsize=10)
    fig_roc.tight_layout()
    fig_roc.savefig(FIG / "fig5_roc_curves.png", dpi=150)
    plt.close(fig_roc)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, m) in zip(axes, metrics.items()):
        ConfusionMatrixDisplay(np.array(m["confusion"]), display_labels=["No default", "Default"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format=","
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Feature importance (Random Forest)
    rf = fitted["Random Forest"]
    feat_names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat_names).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv")
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([n.replace("num__", "").replace("cat__", "") for n in top.index], top.values, color="#4C72B0")
    ax.set_xlabel("Gini importance")
    ax.set_title("Random Forest: top 15 features")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_feature_importance.png", dpi=150)
    plt.close(fig)
    results["top_features"] = [n.replace("num__", "").replace("cat__", "") for n in imp.head(5).index]

    # Baseline: always predict majority class
    metrics["Majority-class baseline"] = {
        "accuracy": float(1 - y_test.mean()), "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5,
    }
    results["metrics"] = metrics

    # Fairness snapshot: recall / false-positive rate by sex and age band (Random Forest)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    eval_df = X_test.copy()
    eval_df["y"] = y_test.values
    eval_df["pred"] = pred
    eval_df["AGE_BAND"] = df.loc[X_test.index, "AGE_BAND"].values
    eval_df["SEX_LABEL"] = eval_df["SEX"].map({1: "Male", 2: "Female"})

    def group_rates(g):
        tp = ((g.pred == 1) & (g.y == 1)).sum()
        fn = ((g.pred == 0) & (g.y == 1)).sum()
        fp = ((g.pred == 1) & (g.y == 0)).sum()
        tn = ((g.pred == 0) & (g.y == 0)).sum()
        return pd.Series({
            "n": len(g),
            "actual_default_rate": g.y.mean(),
            "flag_rate": g.pred.mean(),
            "recall_TPR": tp / max(tp + fn, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
        })

    fair = pd.concat({
        "SEX": eval_df.groupby("SEX_LABEL").apply(group_rates),
        "AGE_BAND": eval_df.groupby("AGE_BAND", observed=True).apply(group_rates),
    }).round(4)
    fair.to_csv(OUT / "fairness_by_group.csv")
    print("\nFairness snapshot (Random Forest, threshold 0.5):\n", fair)
    results["fairness"] = {f"{a}|{b}": v for (a, b), v in fair.to_dict(orient="index").items()}

    # Save metrics table
    pd.DataFrame(metrics).T.drop(columns=["confusion"], errors="ignore").round(4).to_csv(OUT / "model_metrics.csv")


def main() -> None:
    raw = load_raw()
    explore(raw)
    clean = preprocess(raw)
    build_and_evaluate(clean)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved figures ->", FIG)
    print("Saved outputs ->", OUT)


if __name__ == "__main__":
    main()
