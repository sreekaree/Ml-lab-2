import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_excel(r"C:\Users\TRIPU\OneDrive\Documents\ML\Lab Session Data (1).xlsx", sheet_name='IRCTC Stock Price')

price = df['Price']
chg = df['Chg%']

mean_numpy = np.mean(price)
var_numpy = np.var(price)
print(f"Mean (numpy)   : {mean_numpy}")
print(f"Variance (numpy): {var_numpy}")

def custom_mean(data):
    return sum(data) / len(data)

def custom_variance(data):
    m = custom_mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

N_RUNS = 10

t0 = time.perf_counter()
for _ in range(N_RUNS): np.mean(price)
avg_np_mean = (time.perf_counter() - t0) / N_RUNS

t0 = time.perf_counter()
for _ in range(N_RUNS): np.var(price)
avg_np_var = (time.perf_counter() - t0) / N_RUNS

t0 = time.perf_counter()
for _ in range(N_RUNS): custom_mean(price)
avg_cust_mean = (time.perf_counter() - t0) / N_RUNS

t0 = time.perf_counter()
for _ in range(N_RUNS): custom_variance(price)
avg_cust_var = (time.perf_counter() - t0) / N_RUNS

print(f"\nCustom Mean    : {custom_mean(price)}")
print(f"Custom Variance: {custom_variance(price)}")
print(f"\nAccuracy difference (Mean)   : {abs(mean_numpy - custom_mean(price))}")
print(f"Accuracy difference (Variance): {abs(var_numpy - custom_variance(price))}")
print(f"\nAvg time numpy.mean      : {avg_np_mean*1000:.4f} ms")
print(f"Avg time custom_mean     : {avg_cust_mean*1000:.4f} ms")
print(f"Avg time numpy.var       : {avg_np_var*1000:.4f} ms")
print(f"Avg time custom_variance : {avg_cust_var*1000:.4f} ms")

wed_price = df[df['Day'] == 'Wed']['Price']
wed_mean = np.mean(wed_price)
print(f"\nPopulation Mean: {mean_numpy}")
print(f"Wednesday Mean : {wed_mean}")

apr_price = df[df['Month'] == 'Apr']['Price']
apr_mean = np.mean(apr_price)
print(f"\nPopulation Mean: {mean_numpy}")
print(f"April Mean     : {apr_mean}")

is_loss = chg.apply(lambda x: x < 0)
prob_loss = sum(is_loss) / len(chg)
print(f"\nProbability of Loss: {prob_loss}")

wed_chg = df[df['Day'] == 'Wed']['Chg%']
prob_profit_wed = sum(wed_chg > 0) / len(wed_chg)
print(f"\nP(Profit | Wednesday): {prob_profit_wed}")

profit_wed = len(df[(df['Day'] == 'Wed') & (chg > 0)])
prob_wed = len(wed_chg) / len(df)
prob_profit_and_wed = profit_wed / len(df)
cond_prob = prob_profit_and_wed / prob_wed
print(f"\nP(Profit AND Wednesday): {prob_profit_and_wed}")
print(f"P(Wednesday)           : {prob_wed}")
print(f"P(Profit | Wednesday)  : {cond_prob}")

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
df['Day_num'] = df['Day'].map({d: i for i, d in enumerate(day_order)})

plt.figure(figsize=(10, 5))
plt.scatter(df['Day_num'], df['Chg%'], alpha=0.6, edgecolors='black', linewidths=0.5)
plt.xticks(range(len(day_order)), day_order)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Chg% vs Day of the Week')
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    r"C:\Users\TRIPU\OneDrive\Documents\ML\A3_scatter_plot.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nScatter plot saved successfully!")

plt.show()