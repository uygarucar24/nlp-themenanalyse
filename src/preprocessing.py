import re
import spacy

nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])


# Kleinschreibung und Entfernen von URLs und @-Namen
def clean_regex(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return text


# Behält nur relevante Grundformen (keine Füll- und Kurzwörter)
def lemmas_from_doc(doc):
    words = []
    for t in doc:
        lemma = t.lemma_.lower()
        if t.is_alpha and not t.is_stop and len(lemma) >= 3:
            words.append(lemma)
    return " ".join(words)


# Reinigung des gesamten Datensatzes
def clean_corpus(texts):
    cleaned = []
    for t in texts:
        cleaned.append(clean_regex(t))

    docs = []
    for doc in nlp.pipe(cleaned, batch_size=500):
        docs.append(lemmas_from_doc(doc))
    return docs