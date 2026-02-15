import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def main():
    # Load the data extracted from the GPU node
    data = np.load("nim_activations_l13.npz")
    X = data['x']
    y = data['y']

    print(f"Loaded {X.shape[0]} samples with dimension {X.shape[1]}")

    # 1. Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. Plotting
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Label 1 = Cheat (Red), Label 0 = Neutral (Blue)
    scatter = sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=y, 
        palette={1: "#e74c3c", 0: "#3498db"}, 
        alpha=0.6,
        edgecolor='w',
        s=60
    )

    plt.title("Nim Game Activations (Layer 13)\nCheat Pairs vs. Neutral Pairs", fontsize=14)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    
    # Custom Legend
    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles, ["Neutral", "Cheat"], title="Pair Type", loc='best')

    plt.tight_layout()
    plt.savefig("nim_identity_clusters.png", dpi=300)
    print("Plot saved as nim_identity_clusters.png")

if __name__ == "__main__":
    main()