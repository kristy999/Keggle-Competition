import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def inspect_pkl_files(model_path="logreg_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    """
    Loads and inspects the contents of a scikit-learn model and TF-IDF vectorizer
    stored in .pkl files.
    """
    print(f"--- Inspecting {model_path} ---")
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model: {type(model)}")

        if isinstance(model, LogisticRegression):
            print("\nLogistic Regression Model Details:")
            print(f"  Number of features: {model.n_features_in_}")
            print(f"  Classes: {model.classes_}")
            print(f"  Intercept: {model.intercept_}")
            # You can access coefficients, but for a large number of features,
            # printing all of them might be overwhelming.
            # print(f"  Coefficients (first 5): {model.coef_[:, :5]}")
            print("\nTo see more model details, you can access its attributes directly, e.g., model.coef_")
        else:
            print("\nThis model is not a LogisticRegression instance. Further inspection might vary.")

    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred while loading or inspecting the model: {e}")

    print(f"\n--- Inspecting {vectorizer_path} ---")
    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"Successfully loaded vectorizer: {type(vectorizer)}")

        if isinstance(vectorizer, TfidfVectorizer):
            print("\nTF-IDF Vectorizer Details:")
            print(f"  Number of features (unique terms): {len(vectorizer.vocabulary_)}")
            print(f"  Vocabulary (first 20 terms): {list(vectorizer.vocabulary_.keys())[:20]}")
            print(f"  IDF values (for first 20 terms): {vectorizer.idf_[:20]}")
            # You can also get feature names if you want to see them all
            # print(f"  Feature names (first 20): {vectorizer.get_feature_names_out()[:20]}")
            print("\nTo see more vectorizer details, you can access its attributes directly, e.g., vectorizer.vocabulary_")

            # Example: Creating a "table" of top N features and their IDF values
            # This is as close as you'll get to a table for the vectorizer
            print("\nTop 20 Features by TF-IDF Vocabulary Index (simulated table):")
            # Create a list of (feature_name, index, idf_value) tuples
            features_data = []
            for term, idx in vectorizer.vocabulary_.items():
                if idx < len(vectorizer.idf_): # Ensure index is within bounds of idf_ array
                    features_data.append({'Feature': term, 'Index': idx, 'IDF Value': vectorizer.idf_[idx]})

            # Sort by index to see the original order, or by IDF to see important terms
            features_df = pd.DataFrame(features_data).sort_values(by='Index').head(20)
            print(features_df.to_string(index=False)) # Use to_string to avoid truncation

        else:
            print("\nThis vectorizer is not a TfidfVectorizer instance. Further inspection might vary.")

    except FileNotFoundError:
        print(f"Error: Vectorizer file '{vectorizer_path}' not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred while loading or inspecting the vectorizer: {e}")

if __name__ == "__main__":
    # Ensure you have these .pkl files in the same directory as this script,
    # or provide the full path to them.
    inspect_pkl_files()