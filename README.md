# NLP-Themenanalyse von Beschwerdetexten

Worüber beschweren sich Bahnkundinnen und -kunden am häufigsten? Dieses Projekt beantwortet die Frage mit NLP-Methoden in Python, auf Basis des Datensatzes GermEval 2017. Entstanden ist es als Portfolio zum IU-Kurs „Projekt: Data Analysis“ (DLBDSEDA02).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uygarucar24/nlp-themenanalyse/blob/main/analyse.ipynb)

> Ein Klick auf den Badge öffnet `analyse.ipynb` direkt in Google Colab. Dort im Menü „Runtime“ auf „Run all“ klicken, eine lokale Installation ist nicht nötig.

## Überblick

Die Daten werden geladen und auf echte Beschwerden gefiltert, also auf die als relevant und negativ markierten Texte. Danach übernimmt spaCy die Vorverarbeitung. Die sauberen Texte gehen anschließend in zwei Vektorisierungen, Bag-of-Words und TF-IDF, und aus diesen Matrizen extrahieren NMF und LDA jeweils fünf Themen. Zum Schluss werden die Ergebnisse als Wortwolken und Balkendiagramm dargestellt.

Das häufigste Thema sind Verspätungen mit rund 24 % der Beschwerden, das kleinste Streik und Sicherheit mit rund 12 %.

## Projektstruktur

- `analyse.ipynb`: das Präsentations-Notebook mit der kompletten Analyse. Es läuft von oben nach unten durch.
- `main.py`: führt dieselbe Pipeline als Skript aus (`python main.py`).
- `src/`: die einzelnen Analyse-Schritte als Module (`data_loader.py`, `preprocessing.py`, `vectorization.py`, `topic_modeling.py`, `visualization.py`).
- `src/find_n_topics.py`: bestimmt die Themenzahl über die Ellbogen- und die Coherence-Methode.
- `results/`: die erzeugten Abbildungen.

### Warum es das Notebook und die Module gibt

Die Pipeline liegt bewusst zweifach vor. Das Notebook enthält den Code inline und ist dadurch
eigenständig lauffähig, sodass der Colab-Weg ohne Installation und ohne Klonen des Repositorys
funktioniert. `src/` enthält dieselbe Pipeline als wiederverwendbare Module für den Skriptweg
über `main.py`.

Beide Wege liefern identische Ergebnisse, also dieselben 4.863 Beschwerden, dieselben Themen
und dieselben Themengrößen. Sie unterscheiden sich nur in zwei Punkten, die sich aus der
Laufumgebung ergeben. Die Module legen eine lokale Kopie der Daten unter `data/` an und
speichern die Abbildungen nach `results/`. Das Notebook zeigt die Abbildungen inline, weil
eine Colab-Sitzung kein dauerhaftes Dateisystem hat.

## Installation

```bash
pip install -r requirements.txt
```

Das deutsche spaCy-Modell `de_core_news_sm` steht in `requirements.txt` und wird dabei
mitinstalliert. Ein separater `python -m spacy download` ist nicht nötig.

## Ausführen

- In Google Colab, ohne Installation: oben auf den „Open in Colab“-Badge klicken und „Run all“ wählen.
- Lokal im Notebook: `analyse.ipynb` öffnen und alle Zellen ausführen.
- Als Skript, aus dem Projektverzeichnis: `python main.py`
- Bestimmung der Themenzahl (Ellbogen- und Coherence-Methode), ebenfalls aus dem
  Projektverzeichnis: `python src/find_n_topics.py`

## Datenquelle

GermEval 2017, rund 19.400 öffentliche deutschsprachige Beiträge über die Deutsche Bahn. Knapp drei Viertel stammen aus sozialen Netzwerken, überwiegend von Twitter und Facebook, der Rest aus Foren, von Nachrichtenseiten und einzelnen Webseiten.

Eine Kopie der Daten liegt im Repository unter `data/germeval.csv` und wird von dort gelesen. Fehlt sie, etwa beim Start über den Colab-Badge, lädt der Code sie von Hugging Face. Für die Themenanalyse werden nur die als relevant und negativ markierten Texte verwendet, also rund 4.900 echte Beschwerden.
