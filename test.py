import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
plot = sns.jointplot(x="km", y="price", data=df, kind="reg")
plot.savefig("plot.png")
print("Plot saved as: plot.png")
