import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log/000-ddpg-FetchSlide-v1-normal/progress.csv")
plt.plot(df["Episodes"], df["Success"], label="normal")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
# plt.title("Success Rate of HER on FetchSlide Task")

df_hgg = pd.read_csv("log/000-ddpg-FetchSlide-v1-hgg-02-12-2021_09-22-56/progress.csv")
plt.plot(df_hgg["Episodes"], df_hgg["Success"], label="hgg")

plt.legend()
plt.show()