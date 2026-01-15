from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


@st.cache_resource
def load_pipeline():
    return joblib.load(ART / "spam_pipeline.joblib")


@st.cache_data
def load_metrics():
    p = ART / "metrics_models.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def prob_bar(prob_ham: float, prob_spam: float):
    df = pd.DataFrame({"class": ["HAM", "SPAM"], "probability": [prob_ham, prob_spam]})
    st.bar_chart(df.set_index("class"))


st.set_page_config(page_title="Spam NLP Classifier", layout="wide")

st.title("Klasyfikator spamu z wykorzystaniem NLP")
st.caption("Aplikacja demonstracyjna do klasyfikacji wiadomości: SPAM vs HAM")

tab_home, tab_clf, tab_analysis = st.tabs(["Strona główna", "Klasyfikator", "Analiza modelu"])


with tab_home:
    st.subheader("O projekcie")
    st.write(
        """
Aplikacja wykrywa spam w wiadomościach tekstowych na podstawie wcześniej
wytrenowanego modelu uczenia maszynowego. Użytkownik wkleja treść wiadomości,
a system zwraca predykcję klasy (**SPAM** lub **HAM**) wraz z wizualizacją
prawdopodobieństw klas.
"""
    )

    st.subheader("Informacje o zbiorze danych")
    st.write(
        """
Zbiór danych użyty w projekcie zapisany jest w pliku **spam_NLP.csv** i zawiera:
- **MESSAGE** – treść wiadomości tekstowej,
- **CATEGORY** – etykietę klasy (0 = HAM, 1 = SPAM).

Dane obejmują różnorodne wiadomości (reklamowe, informacyjne, techniczne).
Zostały podzielone na zbiór treningowy i testowy z zachowaniem proporcji klas.
"""
    )

    st.subheader("Opis użytych technik")
    st.write(
        """
- **Preprocessing NLP**: czyszczenie tekstu, tokenizacja, usuwanie stop-słów, lematyzacja/stemming (NLTK).
- **Wektoryzacja tekstu**: CountVectorizer oraz TfidfVectorizer.
- **Modele ML**: MultinomialNB, LogisticRegression, SVM (LinearSVC).
- **Prawdopodobieństwa dla SVM**: zastosowano kalibrację (CalibratedClassifierCV), aby uzyskać predict_proba().
- **Tuning hiperparametrów**: RandomizedSearchCV dla TF-IDF + LogisticRegression (analiza wpływu C).
- **Ewaluacja**: accuracy, precision, recall, F1-score oraz macierz pomyłek.
"""
    )

    st.subheader("Najlepszy model (wg wyników testowych)")
    mdf = load_metrics()
    if mdf.empty:
        st.info("Brak `metrics_models.csv`. Uruchom trening, aby zobaczyć wyniki porównania modeli.")
    else:
        best_row = mdf.sort_values("f1", ascending=False).iloc[0]
        st.markdown(
            f"""
- **Najlepszy model:** `{best_row['name']}`  
- **Wektoryzator:** `{best_row['vectorizer']}`  
- **Model:** `{best_row['model']}`  
- **Accuracy:** `{best_row['accuracy']:.4f}`  
- **F1-score:** `{best_row['f1']:.4f}`
"""
        )
        st.caption("W projekcie jako model końcowy wykorzystywany jest model o najwyższym F1-score na zbiorze testowym.")


with tab_clf:
    st.subheader("Sprawdź wiadomość")

    pipeline = None
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(
            "Nie mogę wczytać modelu. Uruchom najpierw trening: "
            "`python -m src.train --data_path spam_NLP.csv`."
        )
        st.exception(e)

    example_spam = "CONGRATULATIONS! You have won a FREE prize. Click here to claim now!"
    example_ham = "As many of you have noticed, the feeds are off now. Will they be back on? I hope so. Soon? I'm not sure. What I will (and can) say is that the experience was very valuable for us"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Wczytaj przykład SPAM"):
            st.session_state["msg"] = example_spam
    with col2:
        if st.button("Wczytaj przykład HAM"):
            st.session_state["msg"] = example_ham

    msg = st.text_area(
        "Wpisz wiadomość:",
        key="msg",
        height=160,
        placeholder="Wklej treść maila...",
    )

    if pipeline is not None:
        if st.button("Klasyfikuj"):
            if not msg.strip():
                st.warning("Wpisz treść wiadomości.")
            else:
                pred = int(pipeline.predict([msg])[0])
                label = "SPAM" if pred == 1 else "HAM"
                st.markdown(f"### Wynik: **{label}**")

                # Prefer probabilities (required in Streamlit spec)
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba([msg])[0]
                    prob_ham, prob_spam = float(proba[0]), float(proba[1])
                    st.write(f"P(HAM) = {prob_ham:.3f} | P(SPAM) = {prob_spam:.3f}")
                    prob_bar(prob_ham, prob_spam)

                # Fallback (should rarely happen now)
                elif hasattr(pipeline, "decision_function"):
                    score = float(pipeline.decision_function([msg])[0])
                    st.write(f"Score (decision function) = {score:.3f}")
                    st.caption("Dodatni score sugeruje klasę SPAM, ujemny sugeruje HAM.")
                else:
                    st.info("Model nie udostępnia ani predict_proba, ani decision_function.")


with tab_analysis:
    st.subheader("Porównanie modeli")
    mdf = load_metrics()
    if mdf.empty:
        st.info("Brak `metrics_models.csv`. Uruchom trening.")
    else:
        st.dataframe(mdf.sort_values("f1", ascending=False), use_container_width=True)

    st.subheader("Wizualizacje modelu")

    cm_path = ART / "confusion_matrix.png"
    plot_path = ART / "metrics_comparison.png"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Macierz pomyłek (model końcowy)**")
        if cm_path.exists():
            st.image(str(cm_path), width=420)
        else:
            st.info("Brak confusion_matrix.png")

    with col2:
        st.markdown("**Porównanie F1-score modeli**")
        if plot_path.exists():
            st.image(str(plot_path), width=420)
        else:
            st.info("Brak metrics_comparison.png")

    st.subheader("Classification report (model końcowy)")
    rep_path = ART / "best_model_report.txt"
    if rep_path.exists():
        st.code(rep_path.read_text(encoding="utf-8"))
    else:
        st.info("Brak best_model_report.txt")
