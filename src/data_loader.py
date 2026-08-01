import os
import pandas as pd

DATA_URL = "https://huggingface.co/datasets/akash418/germeval_2017/resolve/main/train.csv"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CSV = os.path.join(BASE_DIR, "data", "germeval.csv")


# Lädt Daten und speichert als csv Datei 
def load_data(max_docs):
    if os.path.exists(LOCAL_CSV):
        df = pd.read_csv(LOCAL_CSV)
        print(f"Lokal geladen: {len(df)} Dokumente")
    else:
        df = pd.read_csv(DATA_URL)
        os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
        df.to_csv(LOCAL_CSV, index=False)
        print(f"Von Hugging Face geladen und gespeichert: {len(df)} Dokumente")

    df["text"] = df["text"].astype(str).str.strip()
    df = df.drop_duplicates(subset="text")

    # Nur echte Beschwerden behalten: relevant und negatives Sentiment.
    df = df[(df["relevance"] == True) & (df["sentiment"] == "negative")]
    print(f"Nach Beschwerden-Filter (relevant & negativ): {len(df)} Dokumente")

    if max_docs is not None and len(df) > max_docs:
        df = df.sample(max_docs, random_state=24)

    return df.reset_index(drop=True)
