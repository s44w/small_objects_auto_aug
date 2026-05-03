from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "images" / "dota-preliminary-metrics.png"


def main() -> None:
    epochs = list(range(1, 25))
    map50 = [
        0.118,
        0.188,
        0.221,
        0.245,
        0.278,
        0.295,
        0.307,
        0.315,
        0.319,
        0.324,
        0.343,
        0.360,
        0.356,
        0.361,
        0.367,
        0.359,
        0.379,
        0.383,
        0.385,
        0.390,
        0.393,
        0.396,
        0.393,
        0.397,
    ]
    map5095 = [
        0.0668,
        0.105,
        0.125,
        0.144,
        0.160,
        0.170,
        0.181,
        0.185,
        0.189,
        0.192,
        0.203,
        0.213,
        0.214,
        0.216,
        0.219,
        0.210,
        0.227,
        0.227,
        0.232,
        0.239,
        0.238,
        0.239,
        0.238,
        0.243,
    ]

    final_modes = {
        "baseline": {"mAP50": 0.231, "mAP50-95": 0.137},
        "manual": {"mAP50": 0.222, "mAP50-95": 0.133},
        "adaptive": {"mAP50": 0.397, "mAP50-95": 0.244},
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, map50, marker="o", linewidth=2.2, markersize=4.5, label="mAP50")
    ax.plot(epochs, map5095, marker="s", linewidth=2.2, markersize=4.5, label="mAP50-95")
    ax.axhline(final_modes["baseline"]["mAP50-95"], color="#888888", linestyle="--", linewidth=1.2, label="baseline mAP50-95")
    ax.axhline(final_modes["manual"]["mAP50-95"], color="#b56576", linestyle="--", linewidth=1.2, label="manual mAP50-95")
    ax.set_title("Динамика метрик адаптивного режима по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение метрики")
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(0.05, 0.42)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    ax = axes[1]
    names = list(final_modes.keys())
    x = range(len(names))
    width = 0.35
    map50_vals = [final_modes[name]["mAP50"] for name in names]
    map5095_vals = [final_modes[name]["mAP50-95"] for name in names]
    ax.bar([i - width / 2 for i in x], map50_vals, width=width, label="mAP50", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], map5095_vals, width=width, label="mAP50-95", color="#f58518")
    ax.set_title("Итоговое сравнение режимов на DOTA")
    ax.set_xlabel("Режим обучения")
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 0.45)
    ax.legend()

    fig.suptitle("Предварительный эксперимент на датасете DOTA", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
