import pandas as pd
import numpy as np
import os
from statsmodels.miscmodels.ordinal_model import OrderedModel
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns

relative_json_directory = "../reddit3_output"
json_directory = os.path.abspath(relative_json_directory)
file_paths = [
    os.path.join(json_directory, file)
    for file in os.listdir(json_directory)
    if file.endswith(".json")
]
df_list = [pd.read_json(file_path, lines=True) for file_path in file_paths]
df = pd.concat(df_list)

df["daytype"] = df["daytype"].map({"weekend": 1, "weekday": 0})

df["quality"] = pd.Categorical(
    df["quality"], categories=["bad", "normal", "good"], ordered=True
)

formula = "quality ~ sentiment_score + readability_score + daytype - 1"

y, X = dmatrices(formula, df, return_type="dataframe")

y = df["quality"].cat.codes

model = OrderedModel(endog=y, exog=X, distr="logit")

result = model.fit(method="bfgs")

print(result.summary())

params = result.params
conf = result.conf_int()
conf["OR"] = params
conf.columns = ["2.5%", "97.5%", "OR"]
df_results = np.exp(conf)

# # Compute residuals
# residuals = y - result.predict(X)


if not os.path.exists("plots"):
    os.makedirs("plots")

df_results_reversed = df_results.iloc[::-1]
# Coefficient Plot
plt.figure(figsize=(8, 5))
plt.errorbar(
    df_results_reversed["OR"],
    df_results_reversed.index,
    xerr=[
        df_results_reversed["OR"] - df_results_reversed["2.5%"],
        df_results_reversed["97.5%"] - df_results_reversed["OR"],
    ],
    fmt="o",
)
plt.title("Odds Ratios with 95% Confidence Intervals")
plt.xlabel("Odds Ratios")
plt.grid(True)
plt.savefig("plots/coefficient_plot.png")  # Save the figure before showing
plt.show()

# Odds Ratio Plot
plt.figure(figsize=(8, 5))
plt.errorbar(
    df_results_reversed["OR"],
    df_results_reversed.index,
    xerr=[
        df_results_reversed["OR"] - df_results_reversed["2.5%"],
        df_results_reversed["97.5%"] - df_results_reversed["OR"],
    ],
    fmt="o",
)
plt.title("Odds ratio plot with standard errors")
plt.xlabel("Odds Ratios")
plt.xscale("log")
plt.grid(True)
plt.savefig("plots/odds_ratio_plot.png")  # Save the figure before showing
plt.show()

# Pair Plot or Scatter Plot Matrix
pair_plot = sns.pairplot(
    df[["sentiment_score", "readability_score", "daytype", "quality"]], hue="quality"
)
pair_plot.savefig("plots/pair_plot.png")  # Save the figure

# # Residuals Plot
# plt.figure(figsize=(8, 5))
# plt.scatter(result.predict(X), residuals, alpha=0.5)
# plt.title("Residuals vs Fitted Values")
# plt.xlabel("Fitted Values")
# plt.ylabel("Residuals")
# plt.grid(True)
# plt.savefig("plots/residuals_plot.png")  # Save the figure before showing
# plt.show()
