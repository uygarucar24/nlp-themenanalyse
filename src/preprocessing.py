import re
import spacy

nlp = spacy.load("de_core_news_sm", disable=["parser","ner"])


# wandelt den jeweiligen Text in einen bereinigten String um
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    doc = nlp(text)
    words = []
    for t in doc:
        if t.is_alpha and not t.is_stop and len(t.lemma_) >= 3:
            words.append(t.lemma_.lower())
    
    cleaned_text = " ".join(words)
    return cleaned_text


# bereinigt den Datensatz
def clean_corpus(texts):
    docs = []
    for text in texts:
        docs.append(clean_text(text))
    return docs