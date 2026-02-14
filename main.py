import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ─── CONFIG ──────────────────────────────────────────────────────────────
DATA_PATH = "Crop_recommendation.csv"
MODEL_PATH = "crop_model.pkl"
TEST_SIZE = 0.20
RANDOM_SEED = 42
N_ESTIMATORS = 200

# ─── LOAD DATA ───────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("\n🛈 First five rows ↓")
    print(df.head())
    print("\n🛈 Column names & dtypes ↓")
    print(df.dtypes)
    df = df.dropna(subset=['label'])  # remove missing labels if any
    return df

# ─── HEATMAP ─────────────────────────────────────────────────────────────
def show_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.drop(columns=['label'])
    corr = numeric_df.corr(numeric_only=True)
    os.makedirs("static", exist_ok=True)
    heatmap_path = os.path.join("static", "heatmap.png")
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap of Soil Features")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"📊 Heat‑map saved to {heatmap_path}")

# ─── TRAIN MODEL ─────────────────────────────────────────────────────────
def train_random_forest(df: pd.DataFrame) -> float:
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy on test set: {acc:.3%}")

    joblib.dump(clf, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    return acc

# ─── PLOT GENERATOR ──────────────────────────────────────────────────────
def generate_crop_plots(df: pd.DataFrame):
    crops = df['label'].unique()
    os.makedirs("static/plots", exist_ok=True)

    for crop in crops:
        crop_name = crop.strip().lower()
        crop_df = df[df['label'] == crop]

        # 1️⃣ SCATTER PLOT: N vs K
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=crop_df, x='N', y='K')
        plt.title(f"{crop.title()} - N vs K")
        plt.tight_layout()
        plt.savefig(f'static/plots/{crop_name}_scatter.png')
        plt.close()

        # 2️⃣ BAR CHART: Average feature values
        plt.figure(figsize=(6, 4))
        crop_df.mean(numeric_only=True).plot(kind='bar', color='skyblue')
        plt.title(f"{crop.title()} - Average Feature Values")
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(f'static/plots/{crop_name}_bar.png')
        plt.close()

        # 3️⃣ BOX PLOT: pH distribution
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=crop_df['ph'])
        plt.title(f"{crop.title()} - pH Distribution")
        plt.tight_layout()
        plt.savefig(f'static/plots/{crop_name}_box.png')
        plt.close()

    print("✅ Crop-wise plots saved to static/plots/")

# ─── MAIN ENTRY ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    show_heatmap(df)
    train_random_forest(df)
    generate_crop_plots(df)
