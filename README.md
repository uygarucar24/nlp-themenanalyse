# NLP-Themenanalyse von Beschwerdetexten

Automatische Extraktion der häufigsten Themen aus deutschsprachigen Beschwerden
(GermEval 2017, Deutsche Bahn) mit Methoden der maschinellen Sprachverarbeitung (NLP).
Portfolio-Projekt zum IU-Kurs „Projekt: Data Analysis" (DLBDSEDA02).

## Überblick
Workflow: Daten laden → Vorverarbeitung (spaCy) → Vektorisierung (Bag-of-Words & TF-IDF)
→ Themen-Extraktion (NMF & LDA) → Visualisierung (Wortwolken, Themenverteilung).

## Projektstruktur
- `analyse.ipynb` – **Präsentations-Notebook**: die komplette Analyse, läuft eigenständig
  von oben nach unten („Run All").
- `main.py` – führt die gesamte Pipeline als Skript aus (`python main.py`).
- `data_loader.py`, `preprocessing.py`, `vectorization.py`, `topic_modeling.py`,
  `visualization.py` – die einzelnen Analyse-Schritte als Module.
- `find_n_topics.py` – Bestimmung der Themenzahl (Ellbogen- und Coherence-Methode).
- `results/` – erzeugte Abbildungen.

## Installation
```
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

## Ausführen
- **Notebook (empfohlen):** `analyse.ipynb` öffnen und „Run All".
- **Als Skript:** `python main.py`

## Datenquelle
GermEval 2017 – rund 19.400 reale deutschsprachige Beschwerden über die Deutsche Bahn,
zur Laufzeit von Hugging Face geladen.
