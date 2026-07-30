import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from data_loader import load_data
from preprocessing import clean_corpus
from vectorization import build_tfidf



# Ellbogen Methode - Zeichnet den NMF-Rekonstruktionsfehler für verschiedene Werte der Themenanzahl (k)
def plot_elbow(tfidf_matrix, k_min=2, k_max=10):
    """Zeichnet den NMF-Rekonstruktionsfehler für verschiedene Themenzahlen."""
    ks = []
    errors = []
    for k in range(k_min, k_max + 1):
        model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=24)
        model.fit(tfidf_matrix)
        ks.append(k)
        errors.append(model.reconstruction_err_)

    plt.plot(ks, errors, marker="o")
    plt.xlabel("Anzahl Themen (k)")
    plt.ylabel("Rekonstruktionsfehler (kleiner = besser)")
    plt.title("Ellbogen-Methode (NMF)")
    plt.savefig("results/ellbogen.png")
    plt.show()    



def plot_coherence(docs, tfidf_matrix, vectorizer, k_min=2, k_max=10):
    """Berechnet den C_v-Coherence-Score von NMF für verschiedene Themenzahlen."""
    words = vectorizer.get_feature_names_out()

    # gensim benötigt die Texte als Token-Listen und ein Wörterbuch
    texts_tokens = []
    for d in docs:
        texts_tokens.append(d.split())
    dictionary = Dictionary(texts_tokens)

    ks = []
    scores = []
    for k in range(k_min, k_max + 1):
        model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=24)
        model.fit(tfidf_matrix)

        # Top-10-Wörter je Thema sammeln
        topics = []
        for topic in model.components_:
            top_words = []
            for i in topic.argsort()[::-1][:10]:
                top_words.append(words[i])
            topics.append(top_words)

        cm = CoherenceModel(topics=topics, texts=texts_tokens,
                            dictionary=dictionary, coherence="c_v")
        score = cm.get_coherence()
        ks.append(k)
        scores.append(score)
        print(f"k={k}: Coherence={score:.4f}")


if __name__ == "__main__":
    df = load_data(5000)  
    docs = clean_corpus(df["text"])
    tfidf_matrix, tfidf_vectorizer = build_tfidf(docs)
    plot_elbow(tfidf_matrix)
    plot_coherence(docs, tfidf_matrix, tfidf_vectorizer)   