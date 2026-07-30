import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Erstellt je Thema eine Wortwolke und speichert die Abbildung im Ordner results
def show_wordclouds(model, vectorizer, method, n_words=15):
    words = vectorizer.get_feature_names_out()
    n_topics = len(model.components_)

    cols = 3
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.ravel()

    for i, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_words]
        weights = {}
        for idx in top_indices:
            weights[words[idx]] = topic[idx]

        cloud = WordCloud(width=400, height=300, background_color="white",
                          colormap="viridis").generate_from_frequencies(weights)
        axes[i].imshow(cloud, interpolation="bilinear")
        axes[i].set_title(f"{method} - Thema {i + 1}")
        axes[i].axis("off")

    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


