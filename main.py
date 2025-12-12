import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools import exponential, logistic
from tools import get_exponential_params, get_logistic_params

df = pd.read_csv("data.csv")
df["t_year"] = df["year"] - df.iloc[0]["year"]

r_exp, P_init_exp = get_exponential_params(df["t_year"], df["population"])
K, r_log, P_init_log = get_logistic_params(df["t_year"], df["population"])

years = [i for i in range(1850, 2021)]

predicted_populations_exp = [exponential(year - df.iloc[0]["year"], r_exp, P_init_exp) for year in years]
predicted_populations_log = [logistic(year - df.iloc[0]["year"], K, r_log, P_init_log) for year in years]

plt.plot(years, predicted_populations_exp, color="blue", label="exponential")
plt.plot(years, predicted_populations_log, color="green", label="logistic")
plt.scatter(df["year"], df["population"], color="red", label="Census Data")

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Over Time")
plt.legend()
plt.show()
