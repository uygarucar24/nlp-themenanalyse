from sklearn.decomposition import NMF, LatentDirichletAllocation


# Findet Themen mit NMF (auf TF-IDF Matrix, Kullback-Leibler-Divergenz)
def fit_nmf(tfidf_matrix, n_topics, max_iter=200):
    model = NMF(n_components=n_topics, init="nndsvda",
                beta_loss="kullback-leibler", solver="mu",
                max_iter=max_iter, random_state=24)
    model.fit(tfidf_matrix)
    print(f"NMF: {model.n_iter_} von max. {max_iter} Durchläufen")

    return model

# Findet Themen mit LDA (auf Bag of Words Matrix)
def fit_lda(bow_matrix, n_topics, max_iter=200):
    model = LatentDirichletAllocation(n_components=n_topics, learning_method="batch",
                                      max_iter=max_iter, evaluate_every=10, random_state=24)
    model.fit(bow_matrix)
    print(f"LDA: {model.n_iter_} von max. {max_iter} Durchläufen")

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
