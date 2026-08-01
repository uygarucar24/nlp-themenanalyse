from src.data_loader import load_data
from src.preprocessing import clean_corpus
from src.vectorization import build_bow, build_tfidf, compare_vectorizers
from src.topic_modeling import fit_nmf, fit_lda, show_topics
from src.visualization import show_wordclouds, show_topic_sizes

N_TOPICS = 5


def main():
    df = load_data()
    docs = clean_corpus(df["text"])

    bow_matrix, bow_vectorizer = build_bow(docs)
    tfidf_matrix, tfidf_vectorizer = build_tfidf(docs)

    print("=== Vergleich BoW vs. TF-IDF ===")
    compare_vectorizers(bow_matrix, bow_vectorizer, tfidf_matrix, tfidf_vectorizer)

    nmf_model = fit_nmf(tfidf_matrix, N_TOPICS)
    lda_model = fit_lda(bow_matrix, N_TOPICS)

    print("\n=== NMF (auf TF-IDF) ===")
    show_topics(nmf_model, tfidf_vectorizer)

    print("\n=== LDA (auf BoW) ===")
    show_topics(lda_model, bow_vectorizer)

    # Themengrößen je Thema
    show_topic_sizes(nmf_model, tfidf_matrix, "NMF")
    show_topic_sizes(lda_model, bow_matrix, "LDA")

    show_wordclouds(nmf_model, tfidf_vectorizer, "NMF")
    show_wordclouds(lda_model, bow_vectorizer, "LDA")


if __name__ == "__main__":
    main()
