from src.data_loader import load_data
from src.preprocessing import clean_corpus
from src.vectorization import build_bow, build_tfidf
from src.topic_modeling import fit_nmf, fit_lda, show_topics
from src.visualization import show_wordclouds

# zentrale Einstellungen
N_DOCS = 10000
N_TOPICS = 6


def main():
    df = load_data(N_DOCS)
    docs = clean_corpus(df["text"])

    bow_matrix, bow_vectorizer = build_bow(docs)
    tfidf_matrix, tfidf_vectorizer = build_tfidf(docs)

    nmf_model = fit_nmf(tfidf_matrix, n_topics=N_TOPICS)
    lda_model = fit_lda(bow_matrix, n_topics=N_TOPICS)

    print("=== NMF (auf TF-IDF) ===")
    show_topics(nmf_model, tfidf_vectorizer)

    print("\n=== LDA (auf BoW) ===")
    show_topics(lda_model, bow_vectorizer)

    show_wordclouds(nmf_model, tfidf_vectorizer, "NMF")
    show_wordclouds(lda_model, bow_vectorizer, "LDA")


if __name__ == "__main__":
    main()
