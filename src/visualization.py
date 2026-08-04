import os
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def result_path(filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


# Eine Wortwolke je Thema, Schriftgröße entspricht dem Gewicht im Thema
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
                          colormap="viridis", random_state=24).generate_from_frequencies(weights)
        axes[i].imshow(cloud, interpolation="bilinear")
        axes[i].set_title(f"{method} - Thema {i + 1}")
        axes[i].axis("off")

    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(result_path(f"wortwolken_{method}.png"))
    plt.show()


# Ordnet jede Beschwerde ihrem stärksten Thema zu und zeigt die Themengrößen als Balkendiagramm
def show_topic_sizes(model, matrix, method):
    doc_topic = model.transform(matrix)
    n_topics = len(model.components_)

    counts = np.zeros(n_topics, dtype=int)
    for row in doc_topic:
        best = row.argmax()
        counts[best] = counts[best] + 1

    total = counts.sum()
    order = counts.argsort()[::-1]

    labels = []
    sizes = []
    for i in order:
        labels.append(f"Thema {i + 1}")
        sizes.append(counts[i])

    plt.figure(figsize=(8, 5))
    positions = range(len(labels))
    plt.barh(positions, sizes, color="#4C72B0")
    plt.yticks(positions, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Anzahl Dokumente")
    plt.title(f"Themengrößen ({method})")
    for i in positions:
        pct = 100 * sizes[i] / total
        plt.text(sizes[i], i, f" {pct:.1f}%", va="center")

    plt.tight_layout()
    plt.savefig(result_path(f"themengrößen_{method}.png"))
    plt.show()

    print(f"\nThemengrößen ({method}):")
    for i in order:
        pct = 100 * counts[i] / total
        print(f"  Thema {i + 1}: {counts[i]} Dokumente ({pct:.1f}%)")
