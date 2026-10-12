from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "reports" / "tables"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Credit Risk Model Validation",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1280px; padding-top: 2rem;}
        [data-testid="stMetric"] {
            background: #f7f9fc;
            border: 1px solid #e4e9f1;
            border-radius: 12px;
            padding: 14px 16px;
        }
        .small-note {
            color: #5f6b7a;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


@st.cache_data
def load_project_data():
    required_files = {
        "predictions": TABLE_DIR / "test_predictions.csv",
        "ranking": TABLE_DIR / "test_ranking_metrics.csv",
        "calibration": TABLE_DIR / "calibration_summary.csv",
        "shap": TABLE_DIR / "shap_feature_importance.csv",
        "model_comparison": TABLE_DIR / "model_comparison.csv",
        "sensitive": TABLE_DIR / "sensitive_feature_analysis.csv",
        "sex_audit": TABLE_DIR / "sex_subgroup_audit.csv",
        "education_audit": TABLE_DIR / "education_subgroup_audit.csv",
        "marital_audit": TABLE_DIR / "marital_subgroup_audit.csv",
        "metadata": MODEL_DIR / "model_metadata.json",
    }

    missing = [
        str(path.relative_to(BASE_DIR))
        for path in required_files.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required project files: " + ", ".join(missing)
        )

    with open(required_files["metadata"], encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "predictions": pd.read_csv(required_files["predictions"]),
        "ranking": pd.read_csv(required_files["ranking"], index_col=0),
        "calibration": pd.read_csv(required_files["calibration"]),
        "shap": pd.read_csv(required_files["shap"]),
        "model_comparison": pd.read_csv(
            required_files["model_comparison"], index_col=0
        ),
        "sensitive": pd.read_csv(
            required_files["sensitive"], index_col=0
        ),
        "sex_audit": pd.read_csv(required_files["sex_audit"]),
        "education_audit": pd.read_csv(
            required_files["education_audit"]
        ),
        "marital_audit": pd.read_csv(
            required_files["marital_audit"]
        ),
        "metadata": metadata,
    }


def calculate_metrics(actual, probability, threshold, fn_cost, fp_cost):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)

    return {
        "threshold": threshold,
        "accuracy": safe_divide(tp + tn, len(actual)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "high_risk_rate": predicted.mean(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "cost": int(fp_cost * fp + fn_cost * fn),
    }


def build_threshold_grid(actual, probability, fn_cost, fp_cost):
    rows = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        rows.append(
            calculate_metrics(
                actual,
                probability,
                round(float(threshold), 2),
                fn_cost,
                fp_cost,
            )
        )
    return pd.DataFrame(rows)


def percentage(value):
    return f"{value:.1%}"


try:
    project = load_project_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Place app.py in the project root beside the models, reports, "
        "and figures folders, then rerun Streamlit."
    )
    st.stop()


predictions = project["predictions"]
actual = predictions["actual_default"].astype(int).to_numpy()
probability = predictions["predicted_probability"].to_numpy()
metadata = project["metadata"]
development_threshold = float(metadata.get("selected_threshold", 0.19))

st.title("Credit Default Risk — Model Validation Dashboard")
st.caption(
    "Tuned XGBoost benchmark · untouched holdout sample · "
    "cost-sensitive operating threshold"
)

with st.sidebar:
    st.header("Operating assumptions")
    selected_threshold = st.slider(
        "Decision threshold",
        min_value=0.01,
        max_value=0.99,
        value=development_threshold,
        step=0.01,
        help=(
            "Customers at or above this predicted probability are "
            "classified as high risk."
        ),
    )
    false_negative_cost = st.number_input(
        "False-negative cost",
        min_value=1,
        max_value=100,
        value=int(metadata.get("false_negative_cost", 5)),
        step=1,
    )
    false_positive_cost = st.number_input(
        "False-positive cost",
        min_value=1,
        max_value=100,
        value=int(metadata.get("false_positive_cost", 1)),
        step=1,
    )
    st.divider()
    st.markdown(
        f"**Development-selected threshold:** "
        f"`{development_threshold:.2f}`"
    )
    st.caption(
        "The controls are scenario-analysis tools. They do not retune "
        "the validated operating threshold on the holdout sample."
    )


current = calculate_metrics(
    actual,
    probability,
    selected_threshold,
    false_negative_cost,
    false_positive_cost,
)
threshold_grid = build_threshold_grid(
    actual,
    probability,
    false_negative_cost,
    false_positive_cost,
)

fpr, tpr, _ = roc_curve(actual, probability)
global_metrics = {
    "ROC-AUC": roc_auc_score(actual, probability),
    "PR-AUC": average_precision_score(actual, probability),
    "KS": np.max(tpr - fpr),
    "Brier Score": brier_score_loss(actual, probability),
}

overview_tab, threshold_tab, interpretation_tab, governance_tab = st.tabs(
    [
        "Overview",
        "Threshold Simulator",
        "Calibration & SHAP",
        "Governance",
    ]
)


with overview_tab:
    st.subheader("Holdout ranking and probability performance")
    columns = st.columns(4)
    for column, (metric, value) in zip(columns, global_metrics.items()):
        column.metric(metric, f"{value:.4f}")

    st.subheader(f"Operating performance at threshold {selected_threshold:.2f}")
    columns = st.columns(5)
    columns[0].metric("Recall", percentage(current["recall"]))
    columns[1].metric("Precision", percentage(current["precision"]))
    columns[2].metric("Specificity", percentage(current["specificity"]))
    columns[3].metric(
        "Balanced accuracy", percentage(current["balanced_accuracy"])
    )
    columns[4].metric("High-risk rate", percentage(current["high_risk_rate"]))

    st.markdown("#### Development model comparison")
    comparison = (
        project["model_comparison"]
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    comparison_long = comparison.melt(
        id_vars="metric",
        value_vars=["Logistic Regression", "XGBoost"],
        var_name="model",
        value_name="score",
    )
    comparison_figure = px.bar(
        comparison_long,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        color_discrete_map={
            "Logistic Regression": "#8DA0CB",
            "XGBoost": "#1F77B4",
        },
    )
    comparison_figure.update_layout(
        yaxis_title="Cross-validated score",
        xaxis_title=None,
        legend_title=None,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(comparison_figure, use_container_width=True)
    st.caption(
        "Cross-validation values are development estimates. Final model "
        "performance is represented by the holdout metrics above."
    )


with threshold_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Illustrative cost", f"{current['cost']:,}")
    metric_columns[1].metric(
        "False negatives", f"{current['false_negative']:,}"
    )
    metric_columns[2].metric(
        "False positives", f"{current['false_positive']:,}"
    )
    metric_columns[3].metric(
        "Predicted high risk", percentage(current["high_risk_rate"])
    )

    left, right = st.columns([0.9, 1.1])
    with left:
        confusion_values = [
            [current["true_negative"], current["false_positive"]],
            [current["false_negative"], current["true_positive"]],
        ]
        confusion_figure = go.Figure(
            data=go.Heatmap(
                z=confusion_values,
                x=["Predicted non-default", "Predicted default"],
                y=["Actual non-default", "Actual default"],
                colorscale="Blues",
                showscale=False,
                text=confusion_values,
                texttemplate="%{text:,}",
                textfont={"size": 16},
            )
        )
        confusion_figure.update_layout(
            title="Confusion matrix",
            margin=dict(t=55, l=20, r=20, b=20),
            height=390,
        )
        confusion_figure.update_yaxes(autorange="reversed")
        st.plotly_chart(confusion_figure, use_container_width=True)

    with right:
        cost_figure = px.line(
            threshold_grid,
            x="threshold",
            y="cost",
            title="Illustrative error cost across thresholds",
        )
        cost_figure.add_vline(
            x=development_threshold,
            line_dash="dash",
            line_color="#D62728",
            annotation_text="Development-selected 0.19",
        )
        cost_figure.add_vline(
            x=selected_threshold,
            line_dash="dot",
            line_color="#2CA02C",
            annotation_text="Current scenario",
        )
        cost_figure.update_layout(
            xaxis_title="Decision threshold",
            yaxis_title="FP cost + FN cost",
            margin=dict(t=55, b=20),
            height=390,
        )
        st.plotly_chart(cost_figure, use_container_width=True)

    rate_data = threshold_grid.melt(
        id_vars="threshold",
        value_vars=["precision", "recall", "specificity"],
        var_name="metric",
        value_name="value",
    )
    rate_figure = px.line(
        rate_data,
        x="threshold",
        y="value",
        color="metric",
        title="Classification trade-offs",
    )
    rate_figure.add_vline(
        x=selected_threshold,
        line_dash="dot",
        line_color="#222222",
    )
    rate_figure.update_layout(
        xaxis_title="Decision threshold",
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        legend_title=None,
    )
    st.plotly_chart(rate_figure, use_container_width=True)
    st.warning(
        "Do not choose a new production threshold from this holdout chart. "
        "The validated threshold was selected from development out-of-fold "
        "predictions before the holdout sample was evaluated."
    )


with interpretation_tab:
    left, right = st.columns(2)

    with left:
        calibration = project["calibration"].sort_values(
            "mean_predicted_probability"
        )
        calibration_figure = go.Figure()
        calibration_figure.add_trace(
            go.Scatter(
                x=calibration["mean_predicted_probability"],
                y=calibration["observed_default_rate"],
                mode="lines+markers",
                name="Tuned XGBoost",
                line={"color": "#1F77B4", "width": 3},
            )
        )
        calibration_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line={"color": "#777777", "dash": "dash"},
            )
        )
        calibration_figure.update_layout(
            title="Probability calibration",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed default rate",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            legend_title=None,
        )
        st.plotly_chart(calibration_figure, use_container_width=True)
        st.caption(
            f"Brier score: {global_metrics['Brier Score']:.4f}. "
            "Points show ten equal-frequency risk groups."
        )

    with right:
        shap_importance = project["shap"].head(15).sort_values(
            "mean_absolute_shap"
        )
        shap_figure = px.bar(
            shap_importance,
            x="mean_absolute_shap",
            y="feature",
            orientation="h",
            title="Global SHAP importance",
            color="mean_absolute_shap",
            color_continuous_scale="Blues",
        )
        shap_figure.update_layout(
            xaxis_title="Mean absolute SHAP value",
            yaxis_title=None,
            coloraxis_showscale=False,
        )
        st.plotly_chart(shap_figure, use_container_width=True)
        st.caption(
            "Importance measures predictive contribution, not causality. "
            "SHAP values were computed using the model-faithful sparse input."
        )


with governance_tab:
    st.subheader("Model performance without demographic variables")
    sensitive = project["sensitive"].copy()
    st.dataframe(
        sensitive.style.format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(
        "Removing sex, education, and marital status produced only a small "
        "change in development performance. The restricted model requires "
        "independent validation before production consideration."
    )

    st.subheader("Subgroup performance at the validated threshold")
    audit_choice = st.selectbox(
        "Audit variable",
        ["Sex", "Education", "Marital status"],
    )
    audit_lookup = {
        "Sex": project["sex_audit"],
        "Education": project["education_audit"],
        "Marital status": project["marital_audit"],
    }
    audit = audit_lookup[audit_choice].copy()

    rate_columns = [
        "observed_default_rate",
        "high_risk_rate",
        "recall_tpr",
        "false_positive_rate",
        "precision",
    ]
    audit_display = audit.copy()
    for column in rate_columns:
        if column in audit_display:
            audit_display[column] = audit_display[column].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            )
    st.dataframe(audit_display, use_container_width=True, hide_index=True)
    st.info(
        "These are descriptive subgroup diagnostics. Small groups have high "
        "uncertainty, and differences are not evidence of causality or a "
        "complete legal-fairness assessment."
    )


st.divider()
st.caption(
    "Portfolio demonstration using the public UCI Default of Credit Card "
    "Clients dataset. Business costs are illustrative."
)
