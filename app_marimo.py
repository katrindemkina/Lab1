import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from src.sprint2_generators import inverse_cdf
    return mo, pd, np, plt, inverse_cdf


@app.cell
def _(mo):
    amount = mo.ui.slider(start=100, stop=10000, step=100, value=1000,
                          label="Количество сгенерированных значений")
    amount
    return (amount,)


@app.cell
def _(amount, inverse_cdf, mo, np, pd, plt):
    df = pd.read_csv("data/layout.csv")
    counts = df.loc[df["font"] == "DejaVu Serif", "lines"].value_counts().sort_index()
    values = counts.index.to_numpy()
    p = counts.to_numpy() / counts.sum()
    generated = inverse_cdf(values, p, amount.value, seed=42)
    real_frequency = p
    generated_frequency = np.array([(generated == x).mean() for x in values])
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, real_frequency, width=0.36, label="Данные", color="#546C9E")
    ax.bar(x + 0.18, generated_frequency, width=0.36, label="Генератор", color="#D59A75")
    ax.set_xticks(x, values)
    ax.set_xlabel("Количество строк")
    ax.set_ylabel("Доля")
    ax.legend()
    mo.vstack([mo.md(f"**Выборка:** {amount.value} значений"), fig])
    return


if __name__ == "__main__":
    app.run()
