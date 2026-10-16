"""Step 6d: TRAIN a router, and compare it with the hand-written rules.

The task: given a question, predict "sql" (find all / how many) or "search" (find examples).
  1. Data:  data/router/questions.csv  (question, label)
  2. Split: 75% train / 25% test, stratified (same sql/search mix in both). The model never sees test questions.
  3. Model: TF-IDF features (words and word pairs, weighted by how distinctive they are)
            + logistic regression (learns one weight per feature; positive = pushes toward sql).
  4. Compare on the SAME test questions: rules (Step 6c) vs trained model.
  5. Show what the model learned (its strongest words), and every misroute.
"""
import csv
import importlib.util
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.modules.setdefault("ollama", type(sys)("ollama"))      # the router rules don't need the model
spec = importlib.util.spec_from_file_location("query", "scripts/06_query.py")
Q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(Q)

rows = list(csv.DictReader(open("data/router/questions.csv", encoding="utf-8")))
# Trim stray spaces: "sql" and " sql" must be the same label (a labeling slip, not a new category).
X = [r["question"].strip() for r in rows]
y = [r["label"].strip().lower() for r in rows]
bad = sorted({l for l in y if l not in ("sql", "search")})
if bad:
    raise SystemExit(f"Unknown labels {bad}: every label must be sql or search")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"{len(X)} labeled questions -> {len(X_train)} to train on, {len(X_test)} held out for testing\n")

# ---------- Train ----------
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), lowercase=True),   # features: single words AND word pairs ("how many")
    LogisticRegression(max_iter=1000, C=5.0),              # C: how freely it fits the training data
)
model.fit(X_train, y_train)

# ---------- Evaluate both on the same held-out questions ----------
def accuracy(preds):
    return sum(p == t for p, t in zip(preds, y_test)) / len(y_test)

rule_preds = [Q.route(q)[0] for q in X_test]
model_preds = list(model.predict(X_test))
print(f"{'router':16} accuracy   sql questions sent to search (the costly miss)")
for name, preds in [("hand rules", rule_preds), ("trained model", model_preds)]:
    costly = sum(p == "search" and t == "sql" for p, t in zip(preds, y_test))
    print(f"{name:16} {accuracy(preds):7.0%}   {costly}")

# ---------- What did the model learn? ----------
vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
weights = sorted(zip(clf.coef_[0], vec.get_feature_names_out()))
sql_is_positive = clf.classes_[1] == "sql"
toward_sql = weights[::-1][:12] if sql_is_positive else weights[:12]
toward_search = weights[:12] if sql_is_positive else weights[::-1][:12]
print("\nWords pushing toward SQL:   ", ", ".join(w for _, w in toward_sql))
print("Words pushing toward search:", ", ".join(w for _, w in toward_search))

# ---------- Every mistake, from either router ----------
print("\nMisroutes on the test questions:")
for q, t, r, m in zip(X_test, y_test, rule_preds, model_preds):
    if r != t or m != t:
        print(f"  truth {t:6} | rules {r:6} | model {m:6} | {q}")

joblib.dump(model, "data/router/router_model.joblib")
print("\nSaved the trained model to data/router/router_model.joblib")
