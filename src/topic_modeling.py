from sklearn.decomposition import NMF, LatentDirichletAllocation


# Findet Themen mit NMF (auf TF-IDF Matrix, Kullback-Leibler-Divergenz)
def fit_nmf(tfidf_matrix, n_topics):
    model = NMF(n_components=n_topics, init="nndsvda",
                beta_loss="kullback-leibler", solver="mu",
                max_iter=500, random_state=24)
    model.fit(tfidf_matrix)

    return model

# Findet Themen mit LDA (auf Bag of Words Matrix)
def fit_lda(bow_matrix, n_topics):
    model = LatentDirichletAllocation(n_components=n_topics, learning_method="batch", max_iter=50, random_state=24)
    model.fit(bow_matrix)

    return model

# Gibt für jedes Thema die wichtigsten Wörter aus
def show_topics(model, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    for topic_index, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_words]
        top_words = []
        for i in top_indices:
            top_words.append(words[i])
        print(f"Thema {topic_index + 1}:", ", ".join(top_words))
