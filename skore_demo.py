# %% [markdown]
# ## Fetch the dataset
# %%
import skrub
from fairlearn.datasets import fetch_acs_income

df = (
    fetch_acs_income(as_frame=True)
    .frame.sample(10_000, random_state=42, axis="index")
    .reset_index(drop=True)
)
skrub.TableReport(df)
# %% [markdown]
# ## Create a binary classification task that proxies default risk
# %%
X = df.drop(columns=["PINCP"])
y = (df["PINCP"] >= 10_000).astype(int)

# %% [markdown]
# ## Evaluate a logistic regression model, detect problems with skore and use skrub for pre-processing
# %%
import skore
from sklearn.linear_model import LogisticRegression

logistic_report = skore.evaluate(
    LogisticRegression(random_state=0), X, y, splitter=0.2, pos_label=1
)
logistic_report.diagnose()

# %%
y.value_counts()
# %%
preprocessed_estimator = skrub.tabular_pipeline(LogisticRegression(random_state=0))
preprocessed_estimator

# %%
preprocessed_logistic_report = skore.evaluate(
    preprocessed_estimator, X, y, splitter=0.2, pos_label=1
)
preprocessed_logistic_report

# %%
preprocessed_logistic_report.diagnose()

# %%
skore.configuration.ignore_checks = ["SKD004"]

# %% [markdown]
# ## Define a custom check to flag models that do not respect a business requirement on fairness

# %%
import numpy as np
from skore import Check, CheckNotApplicable
from fairlearn.metrics import selection_rate


class CheckFairness(Check):
    code = "CTM001"
    title = "Unfair model"
    report_type = "estimator"
    severity = "issue"
    docs_url = (
        "https://fairlearn.org/v0.13/user_guide/fairness_in_machine_learning.html"
    )

    def __init__(self, sensitive_feature: str):
        self.sensitive_feature = sensitive_feature

    def check_function(self, report):
        "Flag when the model is not fair between groups."
        if self.sensitive_feature not in report.X_test.columns:
            return CheckNotApplicable(self)

        rejection_rates = dict()
        for group in report.X_test[self.sensitive_feature].unique():
            group_mask = report.X_test[self.sensitive_feature] == group
            group_predictions = report.get_predictions(data_source="test")[group_mask]
            group_actual = report.y_test[group_mask]
            rejection_rates[group] = selection_rate(
                group_actual, group_predictions, pos_label=0
            )

        if (
            np.max(list(rejection_rates.values()))
            - np.min(list(rejection_rates.values()))
            > 0.01
        ):
            return (
                f"The model is not fair between groups of different {self.sensitive_feature}. "
                "Rejection rates for groups are: "
                f"{' '.join([f'(group {int(group)}: {rate:.2f})' for group, rate in rejection_rates.items()])}"
            )

        return None


# %%
preprocessed_logistic_report.add_checks([CheckFairness("SEX")])
preprocessed_logistic_report.diagnose()

# %% [markdown]
# ## Define a business metric taking into account the imbalanced costs of misclassification
# %%
from sklearn.metrics import confusion_matrix, make_scorer


def fp_penalty_metric(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])

    gain_matrix = np.array(
        [
            [2, -10],  # Hard penalty on false positives
            [-1, 2],
        ]
    )
    return np.sum(cm * gain_matrix)


fp_penalty_scorer = make_scorer(
    fp_penalty_metric, neg_label=0, pos_label=1, response_method="predict"
)

preprocessed_logistic_report.metrics.add(fp_penalty_scorer, name="FP penalty score")
preprocessed_logistic_report

# %% [markdown]
# ## Tune the threshold to maximize the business-driven score
# %%
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_estimator = TunedThresholdClassifierCV(
    estimator=preprocessed_logistic_report.estimator,
    scoring=fp_penalty_scorer,
    store_cv_results=True,
    random_state=0,
)
tuned_threshold_report = skore.evaluate(
    tuned_estimator, X, y, splitter=0.2, pos_label=1
)
tuned_threshold_report.metrics.add(fp_penalty_scorer, name="FP penalty score")
print(
    f"Best threshold for our custom score: {tuned_threshold_report.estimator.best_threshold_:0.2f}"
)

# %%
comparison_report = skore.compare([preprocessed_logistic_report, tuned_threshold_report])
comparison_report
# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
axs.plot(
    tuned_threshold_report.estimator.cv_results_["thresholds"],
    tuned_threshold_report.estimator.cv_results_["scores"],
    color="tab:orange",
)
axs.plot(
    tuned_threshold_report.estimator.best_threshold_,
    tuned_threshold_report.estimator.best_score_,
    "o",
    markersize=10,
    color="tab:orange",
    label=f"Optimal cut-off point for the FP penalty score: {tuned_threshold_report.estimator.best_threshold_:0.2f}",
)
axs.legend()
axs.set_xlabel("Decision threshold")
axs.set_ylabel("FP penalty score")
_ = fig.suptitle("FP penalty score as a function of the decision threshold")

# %% [markdown]
# ## Visualize threshold selection interactively in the hub

# %%
skore.login(mode="hub")
project_hub = skore.Project(name="20260507-demo-workspace/demo-project", mode="hub")
project_hub.put("tuned_threshold_report", tuned_threshold_report)

# %%
