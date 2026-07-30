import pandas as pd

DATA_URL = "https://huggingface.co/datasets/akash418/germeval_2017/resolve/main/train.csv"
def load_data(max_docs):
    try:
        df = pd.read_csv(DATA_URL)
        print(f"GermEval online geladen: {len(df)} Dokumente")

        df["text"] = df["text"].astype(str).str.strip()
        df = df.drop_duplicates(subset="text")    

        if max_docs is not None and len(df) > max_docs:
            df = df.sample(max_docs, random_state=24)    

        return df.reset_index(drop=True)
    except Exception:
        print("Datensatz konnte nicht geladen werden. Probiere es in ein paar Minuten noch einmal..")

    