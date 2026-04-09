# %% [markdown]
#
# ## Getting started with Skore to evaluate machine learning models

# %%
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer(as_frame=True)
X, y = breast_cancer.data, breast_cancer.target
y = breast_cancer.target_names[y]

# %%
import skrub

skrub.set_config(max_plot_columns=100, max_association_columns=100)
skrub.TableReport(breast_cancer.frame)

# %%
from sklearn.linear_model import LogisticRegression

estimator = skrub.tabular_pipeline(LogisticRegression())
estimator

# %%
import skore

report = skore.evaluate(estimator, X, y, splitter=0.25, pos_label="malignant")
report

# %%
report.help()

# %%
from sklearn.model_selection import KFold

splitter = KFold(n_splits=5, shuffle=True, random_state=42)
report = skore.evaluate(estimator, X, y, splitter=splitter, pos_label="malignant")
report

# %%
report.help()

# %%
display = report.data.analyze()
display.help()

# %%
display

# %%
display.frame()

# %%
display.plot(kind="dist", x="mean radius", y="mean texture", hue="Target")

# %%
report.help()

# %%
display = report.metrics.summarize()
display.frame()

# %%
display.frame(aggregate=None)

# %%
display = report.metrics.roc()
display.plot()

# %%
display.frame()

# %%
display = report.inspection.coefficients()
display.plot()

# %%
display.plot(include_intercept=False, select_k=5, sorting_order="ascending")

# %% [markdown]
#
# ## Storing and retrieve your reports

# %% [markdown]
# ### Starting locally

# %%
from pathlib import Path

project_local = skore.Project(
    name="my-project", mode="local", workspace=Path("./skore-artifacts")
)
project_local.put("log_reg", report)

# %%
project_local.summarize()

# %%
project_local.summarize().reports()

# %% [markdown]
# ### Integration with MLflow

# %%
project_mlflow = skore.Project(
    name="my-project", mode="mlflow", tracking_uri="http://127.0.0.1:5000"
)

# %%
project_mlflow.put("log_reg", report)

# %%
project_mlflow.summarize()

# %%
project_mlflow.summarize().reports()

# %% [markdown]
# ### Skore Hub: experiment tracking for data scientists

# %%
skore.login(mode="hub")
project_hub = skore.Project(name="my-workspace/my-project", mode="hub")
project_hub.put("log_reg", report)

# %%
project_hub.summarize()

# %%
retrieved_report = project_hub.get("skore:report:cross-validation:4738")
retrieved_report.help()

# %%
retrieved_report

# %% [markdown]
# ## What about comparison?

# %%
import numpy as np

Cs = np.logspace(-3, 3, 5)

reports = []
for C in Cs:
    report = skore.evaluate(
        skrub.tabular_pipeline(LogisticRegression(C=C)),
        X,
        y,
        splitter=0.2,
        pos_label="malignant",
    )
    project_hub.put(f"LR_C_{C}", report)
    reports.append(report)

# %%
comparison = skore.compare(reports)

# %%
comparison

# %%
comparison.help()

# %%
comparison.inspection.coefficients().frame()

# %%
