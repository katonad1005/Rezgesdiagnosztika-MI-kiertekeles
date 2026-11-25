# A kijelölt gépek méréseit időrendben osztja tanító- és ellenőrzőcsoportokra.

from datetime import date
import numpy as np
import pandas as pd


def terv_beolvasas(path):
    plan = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not {"gepegyseg_utvonal", "szerep"} <= set(plan):
        raise ValueError("A gépterv mezői: gepegyseg_utvonal és szerep.")
    keys = plan.gepegyseg_utvonal.str.strip()
    roles = plan.szerep.str.strip().str.lower()
    if keys.duplicated().any() or keys.eq("").any() or not roles.isin(["celgep", "tanitogep", "kihagy"]).all():
        raise ValueError("A gépterv egyedi gépútvonalakat és celgep / tanitogep / kihagy szerepeket igényel.")
    if not roles.eq("celgep").any():
        raise ValueError("A géptervben jelölj ki legalább egy célgépet.")
    return dict(zip(keys, roles))


def idorendi_minta(tabla, roles, cutoff):
    boundary = date.fromisoformat(cutoff)
    days = pd.to_datetime(tabla["idopont"], errors="raise").dt.date
    role = tabla.gepegyseg_utvonal.map(roles)
    train = role.isin(["celgep", "tanitogep"]) & (days < boundary)
    test = role.eq("celgep") & (days >= boundary)
    for machine in [m for m,r in roles.items() if r == "celgep"]:
        belongs = tabla.gepegyseg_utvonal.eq(machine)
        if not (belongs & train).any() or not (belongs & test).any():
            raise ValueError(f"A célgépen a határnap előtt és onnantól is kell igazolt jelölt: {machine}")
    subset = tabla.loc[train | test].reset_index(drop=True)
    return subset, int((~(train | test)).sum())


def idorendi_felosztas(tabla, cutoff, maximum):
    if not 2 <= maximum <= 10:
        raise ValueError("A CV-körök felső határa 2–10.")
    days = pd.to_datetime(tabla.idopont).to_numpy(dtype="datetime64[D]")
    boundary = np.datetime64(date.fromisoformat(cutoff))
    dev, test = np.flatnonzero(days < boundary), np.flatnonzero(days >= boundary)
    y = tabla.cel_hibas.to_numpy()
    if len(np.unique(y[test])) != 2:
        raise ValueError("A rögzített célgépes tesztidőszakban mindkét igazolt állapot szükséges az összehasonlításhoz.")
    # A belső tanítás a validációt megelőző napokra korlátozódik.
    dates = days[dev]
    # Egy nap mérései együtt maradnak. A lezárt blokkban mindkét állapot szerepel.
    unique = np.unique(dates)
    starts, classes = [0], set()
    for pos, day in enumerate(unique):
        classes.update(y[dev[dates == day]])
        if len(classes) == 2:
            starts.append(pos + 1)
            classes.clear()
    if len(starts) >= 4:
        chunks = np.array_split(starts[1:-1], min(maximum, len(starts)-2))
        boundaries = [unique[chunk[0]] for chunk in chunks]
        folds = []
        for i, start in enumerate(boundaries):
            valid = dates >= start
            if i + 1 < len(boundaries):
                valid &= dates < boundaries[i + 1]
            folds.append((np.flatnonzero(dates < start), np.flatnonzero(valid)))
        return dev, test, folds
    raise ValueError("Kevés elkülöníthető dátum vagy igazolt állapot a legalább két időrendi belső validációhoz.")
