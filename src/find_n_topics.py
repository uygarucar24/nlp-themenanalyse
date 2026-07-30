import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

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
    




if __name__ == "__main__":
    df = load_data(5000)  
    docs = clean_corpus(df["text"])
    tfidf_matrix, _ = build_tfidf(docs)
    plot_elbow(tfidf_matrix)