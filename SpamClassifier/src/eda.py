from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if set(["label", "text"]).issubset(df.columns):
        df = df.rename(columns={"label": "CATEGORY", "text": "MESSAGE"})
        df["CATEGORY"] = df["CATEGORY"].map({"ham": 0, "spam": 1}).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="spam_NLP.csv")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data_path)).copy()

    df["len"] = df["MESSAGE"].astype(str).str.len()
    class_counts = df["CATEGORY"].value_counts().sort_index()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["HAM (0)", "SPAM (1)"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    ax.set_title("Rozkład klas")
    ax.set_ylabel("Liczba wiadomości")
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(df["len"], bins=50)
    ax.set_title("Histogram długości wiadomości")
    ax.set_xlabel("Długość (znaki)")
    ax.set_ylabel("Liczba wiadomości")
    fig.tight_layout()
    fig.savefig(out_dir / "message_length_hist.png", dpi=160)
    plt.close(fig)

    if WordCloud is None:
        ham_text = " ".join(df.loc[df["CATEGORY"] == 0, "MESSAGE"].astype(str).tolist())
        spam_text = " ".join(df.loc[df["CATEGORY"] == 1, "MESSAGE"].astype(str).tolist())
        for name, txt in [("ham", ham_text), ("spam", spam_text)]:
            words = pd.Series(txt.lower().split()).value_counts().head(30)
            words.to_csv(out_dir / f"top_words_{name}.csv")
    else:
        for label, cname in [(0, "ham"), (1, "spam")]:
            text = " ".join(df.loc[df["CATEGORY"] == label, "MESSAGE"].astype(str).tolist())
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"WordCloud: {cname.upper()}")
            fig.tight_layout()
            fig.savefig(out_dir / f"wordcloud_{cname}.png", dpi=160)
            plt.close(fig)

    summary = {
        "n_samples": int(len(df)),
        "n_ham": int(class_counts.get(0, 0)),
        "n_spam": int(class_counts.get(1, 0)),
        "spam_pct": float(class_counts.get(1, 0) / len(df) * 100.0),
        "mean_len": float(df["len"].mean()),
        "median_len": float(df["len"].median()),
    }
    (out_dir / "eda_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")

    print("EDA done. Files saved to:", out_dir)


if __name__ == "__main__":
    main()
