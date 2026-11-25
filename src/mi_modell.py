# Gépenként különíti el az adatokat, és összehasonlítja a véletlen erdő és az SVM eredményét.

from __future__ import annotations

import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tanito_adatok import JELLEMZOK
from fordulatszam import AZONOS_MODSZEREK


FORMATUM = 11


def verziok() -> dict:
    return {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
            'joblib': joblib.__version__}


def _osztalycsoportok(y: np.ndarray, g: np.ndarray) -> list[int]:
    return [len(np.unique(g[y == k])) for k in (0, 1)]


def belso_felosztas(y: np.ndarray, g: np.ndarray, maximum: int, seed: int) -> list:
    """Osztályok és csoportok alapján keres; modellpontszámot nem használ."""
    felso = min(maximum, len(np.unique(g)), min(_osztalycsoportok(y, g)))
    for korok in range(felso, 1, -1):
        for eltol in range(10):
            cv = StratifiedGroupKFold(korok, shuffle=True, random_state=seed + eltol)
            reszek = list(cv.split(np.zeros((len(y), 1)), y, g))
            if all(len(np.unique(y[a])) == len(np.unique(y[b])) == 2 and
                   not set(g[a]) & set(g[b]) for a, b in reszek):
                return reszek
    raise ValueError('Nem képezhető legalább két, mindkét állapotot tartalmazó belső csoportos kör.')


def felosztas(y: np.ndarray, g: np.ndarray, teszt_arany: float = 0.2,
             cv_max: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list]:
    if not 0 < teszt_arany < 1 or not 2 <= cv_max <= 10 or seed < 0:
        raise ValueError('0–1 közötti tesztarány, 2–10 CV-kör és nem negatív seed szükséges.')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Mindkét igazolt állapotból kell tanítási adat: megfelelő (0) és hibás (1).')
    if min(_osztalycsoportok(y, g)) < 3:
        raise ValueError('Ehhez a teszt+CV eljáráshoz mindkét állapotnak legalább három külön '
                         'csoportban kell szerepelnie. Ez technikai minimum, nem elegendőségi igazolás.')
    jeloltek = []
    cv = GroupShuffleSplit(n_splits=128, test_size=teszt_arany, random_state=seed)
    for fejlesztes, teszt in cv.split(np.zeros((len(y), 1)), y, g):
        if len(np.unique(y[teszt])) != 2 or min(_osztalycsoportok(y[fejlesztes], g[fejlesztes])) < 2:
            continue
        elteres = (abs(len(teszt) / len(y) - teszt_arany) +
                   abs(y[teszt].mean() - y.mean()) + abs(y[fejlesztes].mean() - y.mean()))
        jeloltek.append((float(elteres), fejlesztes, teszt))
    for _, fejlesztes, teszt in sorted(jeloltek, key=lambda t: t[0]):
        try:
            belso = belso_felosztas(y[fejlesztes], g[fejlesztes], cv_max, seed)
            return fejlesztes, teszt, belso
        except ValueError:
            continue
    raise ValueError('Nem sikerült elkülönített tesztet és érvényes belső CV-t kialakítani. '
                     'Több vagy megfelelőbben elkülöníthető igazolt mérési eset szükséges.')


def modellek(seed: int = 42) -> dict:
    def potlas():
        return SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    return {
        'RandomForest': (Pipeline([
            ('potlas', potlas()),
            ('osztalyozo', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=seed, n_jobs=1)),
        ]), {'osztalyozo__max_depth': [None, 5], 'osztalyozo__min_samples_leaf': [1, 2, 4]}),
        'SVM': (Pipeline([
            ('potlas', potlas()), ('skalazas', StandardScaler()),
            ('osztalyozo', SVC(kernel='rbf', class_weight='balanced')),
        ]), {'osztalyozo__C': [0.1, 1., 10.], 'osztalyozo__gamma': ['scale', 0.1]}),
    }


def mutatok(y, becsult, suly=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, becsult, labels=[0, 1], sample_weight=suly).ravel()
    return {
        'accuracy': float(accuracy_score(y, becsult, sample_weight=suly)),
        'balanced_accuracy': float(balanced_accuracy_score(y, becsult, sample_weight=suly)),
        'precision': float(precision_score(y, becsult, zero_division=0, sample_weight=suly)),
        'recall': float(recall_score(y, becsult, zero_division=0, sample_weight=suly)),
        'f1': float(f1_score(y, becsult, zero_division=0, sample_weight=suly)),
        'TN': float(tn), 'FP': float(fp), 'FN': float(fn), 'TP': float(tp),
    }


def tanitas(tabla: pd.DataFrame, csoportositas: str = 'gep_nap', teszt_arany: float = 0.2,
            cv_max: int = 5, seed: int = 42, szolas=print, *, teszt_kezdete=None) -> dict:
    if csoportositas not in {'gep_nap', 'gep', 'idorend'}:
        raise ValueError('A csoportosítás gep_nap vagy gep legyen.')
    kell = set(JELLEMZOK + ['rekord_id', 'cel_hibas', 'cimke_statusz', 'tanitasi_jelolt',
                          'gepegyseg_utvonal', 'felosztasi_csoport', 'harmonikus_tures_rel',
                          'mertekegyseg', 'detektalas', 'fordulat_modszer'])
    if kell - set(tabla):
        raise ValueError(f'Hiányos közös adatkészlet: {sorted(kell - set(tabla))}')
    if tabla.empty:
        raise ValueError('Nincs igazolt, felhasználható tanítási rekord. Az adatkeszlet_jelentes.json '
                         'és a címkék ellenőrzése szükséges.')
    if not tabla.fordulat_modszer.isin(AZONOS_MODSZEREK).all():
        raise ValueError('Eltérő vagy elavult fordulatmódszer. Futtasd újra a nyers exportok jellemzőkinyerését az 5–100 Hz eljárással.')
    if not tabla.cimke_statusz.eq('igazolt_rekordcimke').all() or not tabla.tanitasi_jelolt.eq(True).all():
        raise ValueError('Csak az ellenőrzött, igazolt állapotú tanítási jelöltek használhatók.')
    if tabla.rekord_id.isna().any() or tabla.rekord_id.astype(str).str.strip().eq('').any() or tabla.rekord_id.duplicated().any():
        raise ValueError('Hiányzó vagy ismétlődő rekordazonosító a tanítótáblában.')
    if not tabla.cel_hibas.isin([0, 1]).all():
        raise ValueError('Hiányzó vagy nem bináris célváltozó.')
    X = tabla[JELLEMZOK].astype(float)
    if np.isinf(X.to_numpy()).any() or X.lt(0).any().any() or X.isna().all(axis=1).any():
        raise ValueError('Érvénytelen modelljellemző. A hiányzó érték maradhat üres, de nem minden jellemző egyszerre.')
    toleranciak = pd.to_numeric(tabla.harmonikus_tures_rel, errors='coerce')
    if toleranciak.isna().any() or toleranciak.nunique() != 1 or not 0 <= toleranciak.iloc[0] < 0.25:
        raise ValueError('Egyetlen ismert harmonikussáv-beállítás szükséges. A régi tábla mellett '
                         'legyen meg az abra_beallitasok.json, vagy futtasd újra a jellemzőkinyerést.')
    if tabla.mertekegyseg.nunique() != 1 or tabla.detektalas.nunique() != 1:
        raise ValueError('Eltérő mértékegység vagy detektálás nem keverhető ebben a modellben.')
    y = tabla.cel_hibas.to_numpy(dtype=int)
    kulcs = 'felosztasi_csoport' if csoportositas != 'gep' else 'gepegyseg_utvonal'
    if tabla[kulcs].isna().any() or tabla[kulcs].astype(str).str.strip().eq('').any():
        raise ValueError('Hiányzó felosztási csoport.')
    g = tabla[kulcs].astype(str).to_numpy()
    if csoportositas == "idorend":
        from mi_idorend import idorendi_felosztas
        fejlesztes, teszt, belso = idorendi_felosztas(tabla, teszt_kezdete, cv_max)
    else:
        fejlesztes, teszt, belso = felosztas(y, g, teszt_arany, cv_max, seed)
    szolas(f'Fejlesztés: {len(fejlesztes)} rekord; elkülönített teszt: {len(teszt)} rekord; belső CV: {len(belso)} kör.')
    pontozas = {'balanced_accuracy': 'balanced_accuracy',
                'f1': make_scorer(f1_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0)}
    keresok, ranglista = {}, []
    for nev, (pipeline, racs) in modellek(seed).items():
        szolas(f'Tanítás és belső paraméterkeresés: {nev}...')
        kereso = GridSearchCV(pipeline, racs, scoring=pontozas, refit='balanced_accuracy',
                             cv=belso, n_jobs=1, error_score='raise', return_train_score=False)
        kereso.fit(X.iloc[fejlesztes], y[fejlesztes])
        keresok[nev] = kereso
        ranglista.append({'modell': nev, 'cv_balanced_accuracy': float(kereso.best_score_),
                          'parameterek': kereso.best_params_})
    
    # A modellválasztás a teszt előtt lezárul. Döntetlennél a véletlen erdő kerül előre.
    valasztott = max(ranglista, key=lambda r: r['cv_balanced_accuracy'])['modell']
    prediktorok = {nev: kereso.best_estimator_ for nev, kereso in keresok.items()}
    prediktorok['Referencia'] = DummyClassifier(strategy='most_frequent').fit(X.iloc[fejlesztes], y[fejlesztes])
    eredmenyek, becslesek = [], {}
    csoport_suly = 1.0 / pd.Series(g[teszt]).map(pd.Series(g[teszt]).value_counts()).to_numpy()
    for nev, modell in prediktorok.items():
        becsult = modell.predict(X.iloc[teszt]).astype(int)
        becslesek[nev] = becsult
        eredmenyek.append({'modell': nev, 'teszt': mutatok(y[teszt], becsult),
                            'csoportsulyozott_teszt': mutatok(y[teszt], becsult, csoport_suly)})
    return {
        'valasztott': valasztott, 'modell': prediktorok[valasztott],
        'fejlesztes_index': fejlesztes, 'teszt_index': teszt, 'belso_cv': belso,
        'csoportok': g, 'eredmenyek': eredmenyek, 'becslesek': becslesek,
        'keresok': keresok, 'ranglista': ranglista,
        'harmonikus_tures_rel': float(toleranciak.iloc[0]),
    }
