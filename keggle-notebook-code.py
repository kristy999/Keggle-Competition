import pandas as pd

# Load the training labels
df = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')

# Preview
df.head()

import os

base_path = '/kaggle/input/make-data-count-finding-data-references'

# List all files/folders under the main dataset folder
for root, dirs, files in os.walk(base_path):
    print(f"\n Folder: {root}")
    for name in files:
        print(f"  📄 {name}")


import pandas as pd
import os

# Set correct paths
data_path = '/kaggle/input/make-data-count-finding-data-references'
xml_dir = f'{data_path}/train/XML'

# Load labels
df = pd.read_csv(f'{data_path}/train_labels.csv')

# List available XMLs
xml_files = os.listdir(xml_dir)
available_xml_ids = [f.replace('.xml', '') for f in xml_files]

# Filter to rows that have a matching XML file
valid_rows = df[df['article_id'].isin(available_xml_ids)]
valid_rows.head()

import xml.etree.ElementTree as ET

article_id = '10.1002_2017jc013030'
dataset_doi = '10.17882/49388'  # Remove prefix for matching

xml_path = f"/kaggle/input/make-data-count-finding-data-references/test/XML/{article_id}.xml"

# Parse the XML
try:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    print(f"Searching for DOI: {dataset_doi}\n")
    matches = []
    for elem in root.iter():
        if elem.text and dataset_doi.lower() in elem.text.lower():
            matches.append(elem.text.strip())

    if matches:
        print("📌 Found mention(s):")
        for match in matches:
            print("➡", match)
    else:
        print("❌ No mention found in this XML.")
except FileNotFoundError:
    print("⚠️ XML file not found for this article.")

def extract_context_label(article_id, dataset_doi, label, xml_folder):
    import xml.etree.ElementTree as ET

    dataset_doi = dataset_doi.replace("https://doi.org/", "").lower()
    xml_path = f"{xml_folder}/{article_id}.xml"

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        contexts = []
        for elem in root.iter():
            if elem.text and dataset_doi in elem.text.lower():
                sentence = elem.text.strip()
                contexts.append({
                    "article_id": article_id,
                    "context": sentence,
                    "dataset_id": f"https://doi.org/{dataset_doi}",
                    "label": label
                })
        return contexts
    except Exception as e:
        print(f"Error reading {article_id}: {e}")
        return []

# Path to your XML folder
xml_folder = '/kaggle/input/make-data-count-finding-data-references/test/XML'

# Extract one example
sample = extract_context_label(
    article_id='10.1002_2017jc013030',
    dataset_doi='https://doi.org/10.17882/49388',
    label='Primary',
    xml_folder=xml_folder
)

# Show it
import pandas as pd
pd.DataFrame(sample)

def build_training_data(df, xml_folder):
    import xml.etree.ElementTree as ET
    dataset = []

    for idx, row in df.iterrows():
        article_id = row['article_id']
        dataset_id = row['dataset_id']
        label = row['type']

        # Skip missing data
        if dataset_id == "Missing" or label == "Missing":
            continue

        # Normalize
        clean_doi = dataset_id.replace("https://doi.org/", "").lower()
        xml_path = f"{xml_folder}/{article_id}.xml"

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for elem in root.iter():
                if elem.text and clean_doi in elem.text.lower():
                    context = elem.text.strip()
                    dataset.append({
                        "article_id": article_id,
                        "context": context,
                        "dataset_id": dataset_id,
                        "label": label
                    })
        except:
            continue  # skip file if not found or unreadable

    return pd.DataFrame(dataset)

# Run on available rows only
available_ids = os.listdir('/kaggle/input/make-data-count-finding-data-references/train/XML')
available_ids = [f.replace('.xml', '') for f in available_ids]

# Filter to valid entries
valid_rows = df[df['article_id'].isin(available_ids)]

# Build training data
train_data = build_training_data(valid_rows, '/kaggle/input/make-data-count-finding-data-references/train/XML')

# Preview
train_data.head()

# Filter short contexts (like raw DOIs)
train_data_cleaned = train_data[train_data['context'].str.split().str.len() >= 5]

print(f"Original rows: {len(train_data)}")
print(f"After filtering: {len(train_data_cleaned)}")
train_data_cleaned.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Inputs
X_texts = train_data_cleaned['context']
y_labels = train_data_cleaned['label']

# Convert context text into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(X_texts)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Inputs
X_texts = train_data_cleaned['context']
y_labels = train_data_cleaned['label']

# Convert context text into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(X_texts)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))

# 1. Setup and Imports
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 2. Load training data
data_path = "/kaggle/input/make-data-count-finding-data-references"
df = pd.read_csv(f"{data_path}/train_labels.csv")

# 3. Use dataset_id text as proxy for context (simplified, works for baseline)
df = df.dropna()
X = df["dataset_id"].astype(str)
y = df["type"]

# 4. Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# 5. Train model (basic logistic regression)
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_vec, y)

# 6. Test prediction setup
test_xml_dir = f"{data_path}/test/XML"
predictions = []
row_id = 0

def extract_mentions(xml_file):
    doi_pattern = re.compile(r'(10\.\d{4,9}/[^\s";]+)', re.IGNORECASE)
    acc_pattern = re.compile(r'\b(GSE\d+|E-\w+-\d+|PRJ\w+\d+|CHEMBL\d+|PDB\s+\w+)\b', re.IGNORECASE)
    mentions = set()
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for elem in root.iter():
            if elem.text:
                text = elem.text.strip()
                mentions.update(doi_pattern.findall(text))
                mentions.update(acc_pattern.findall(text))
    except:
        pass
    return mentions

# 7. Loop through test XMLs
for fname in os.listdir(test_xml_dir):
    if not fname.endswith(".xml"):
        continue
    article_id = fname.replace(".xml", "")
    mentions = extract_mentions(os.path.join(test_xml_dir, fname))
    for m in mentions:
        if m.startswith("10."):
            dataset_id = "https://doi.org/" + m.lower()
        else:
            dataset_id = m.upper().strip()
        context_text = m
        X_test = vectorizer.transform([context_text])
        pred = model.predict(X_test)[0]
        predictions.append({
            "row_id": row_id,
            "article_id": article_id,
            "dataset_id": dataset_id,
            "type": pred
        })
        row_id += 1

# 8. Create submission.csv
submission = pd.DataFrame(predictions)
submission = submission.drop_duplicates(subset=["article_id", "dataset_id", "type"])
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv ready! Preview:")
submission.head()

