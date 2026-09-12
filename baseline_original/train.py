import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv("features.csv")
FEATS = [c for c in df.columns if c not in ("anon_id", "label", "split")]
tr, va = df[df.split == "train"], df[df.split == "val"]
Xtr, ytr = tr[FEATS].values, (tr.label == "synthetic").astype(int).values
Xva, yva = va[FEATS].values, (va.label == "synthetic").astype(int).values
print(f"train {len(ytr)} (synth {ytr.mean():.0%})  |  val {len(yva)} (synth {yva.mean():.0%})")

def report(name, p, y):
    pred = (p > 0.5).astype(int)
    eer_fpr, eer_tpr, _ = roc_curve(y, p)
    eer = eer_fpr[np.nanargmin(np.abs((1 - eer_tpr) - eer_fpr))]
    print(f"\n{name}")
    print(f"  accuracy {accuracy_score(y,pred):.3f}   AUC {roc_auc_score(y,p):.3f}"
          f"   EER {eer:.3f}   Brier {brier_score_loss(y,p):.3f}")
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"  humano ok {tn}/{tn+fp}   sintetico ok {tp}/{tp+fn}")
    return accuracy_score(y, pred)

# --- referencia de una sola feature: latencia mediana ---
lat = va["lat_med"].fillna(tr.lat_med.median()).values
report("Solo latencia mediana (umbral 2.0s)", np.clip((lat - 2.0) / 3 + 0.5, .01, .99), yva)

# --- modelo completo ---
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                     max_leaf_nodes=15, l2_regularization=1.0,
                                     random_state=0)
cvp = cross_val_predict(clf, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        method="predict_proba")[:, 1]
report("HGB - validacion cruzada en train", cvp, ytr)

clf.fit(Xtr, ytr)
p = clf.predict_proba(Xva)[:, 1]
report("HGB - val (hablantes disjuntos)", p, yva)

# --- calibrado, porque confidence puntua ---
cal = CalibratedClassifierCV(clf, method="isotonic", cv=5).fit(Xtr, ytr)
pc = cal.predict_proba(Xva)[:, 1]
report("HGB calibrado - val", pc, yva)

imp = pd.Series(clf.feature_importances_ if hasattr(clf, "feature_importances_")
                else np.zeros(len(FEATS)), index=FEATS)
from sklearn.inspection import permutation_importance
pi = permutation_importance(clf, Xva, yva, n_repeats=20, random_state=0)
top = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False).head(10)
print("\nTop features (permutation importance en val):")
for k, v in top.items(): print(f"  {k:18s} {v:.4f}")

# --- guardar bundle para el endpoint ---
import joblib
final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
    max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
Xall = df[FEATS].values; yall = (df.label == "synthetic").astype(int).values
fin = CalibratedClassifierCV(final, method="isotonic", cv=5).fit(Xall, yall)
joblib.dump({"model": fin, "feats": FEATS,
             "medians": df[FEATS].median().to_dict()}, "model.joblib")
print("\n-> model.joblib guardado (entrenado en train+val)")
