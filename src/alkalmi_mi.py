# A teljes mérési alkalmakból tanítóadatot állít össze, modellt tanít és állapotot becsül.

"""Teljes mérési alkalmak importja, gépenkénti tanítása és becslése."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import csv
import os
import sys
import tempfile
from pathlib import Path

from skf_diagnosztika.skf_alkalmak import sessions
from skf_diagnosztika.skf_beallitasok import load_config, load_point_settings
from skf_diagnosztika.skf_geptipus import machine_map
from skf_diagnosztika.skf_fajlkezeles import read_spectra, write_csv
from skf_diagnosztika.skf_feldolgozas import assess_prepared

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["seb_1x", "seb_2x", "seb_3x", "seb_2_per_1", "gyors_max", "burk_E3_max"]
FORMAT = "alkalom_5_100_v1"


# A két igazolt exporthoz tartozó mérési alkalmak.
GOOD_EXPORTS = {"rgk vi jo_spectra.csv", "rgk_vi_jo_minta_spectra.csv"}
CONFIRMED_GOOD = {
    "57de67d9def7034b9d5c93c9": "RGK VI jo_Spectra.csv",
    "6a57688164c14b7bb925aa55": "RGK_VI_jo_minta_Spectra.csv",
    "79ca2840cb9427538771a0d5": "RGK_VI_jo_minta_Spectra.csv",
    "1794962b40bbab7b961a4d3d": "RGK_VI_jo_minta_Spectra.csv",
    "60359c4309be9cc223dcb75f": "RGK_VI_jo_minta_Spectra.csv",
}


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")


def collect(paths, machine_file, config, points, *, legkezelo_export=False):
    mapping = machine_map(machine_file)
    records = {}
    for path in paths:
        spectra, _, note = read_spectra(path)
        if not spectra:
            raise ValueError(f"{path}: {note or 'üres spektrumexport'}")
        for record in spectra:
            if record.id in records:
                records[record.id].sources.extend(record.sources)
            else:
                records[record.id] = record
    if legkezelo_export:
        # Az új légkezelő egységeit a közvetlen szülőútvonal azonosítja.
        # A közös skála csak az egyező mérésmódokra érvényes.
        expected = {"VT": ("mm/s", "rms"), "VS": ("mm/s", "rms"),
                    "AC": ("g", "peak"), **dict.fromkeys(("E2", "E3", "E4"), ("ge", "peak to peak"))}
        for record in records.values():
            parent, separator, leaf = record.point.rpartition(chr(92))
            if not separator or not parent or not leaf:
                raise ValueError("Hiányos SKF mérésipont-útvonal: " + record.point)
            _, kind = mapping.setdefault(parent, (parent, "legkezelo"))
            mode = leaf.upper().split("-")[-1]
            units = (record.meta["Unit"].casefold(), record.meta["Detection"].casefold())
            if kind == "legkezelo" and expected.get(mode) == units:
                defaults = dict(frekvenciaegyseg="Hz", kezdo_frekvencia_hz=0,
                                vegpontok="ket_vegpont", skala_ellenorzott=True)
                points[record.point] = {**defaults, **points.get(record.point, {})}
    prepared = [(r,None,None,"",None) for r in records.values()]
    _, results = assess_prepared(prepared, config, points, supplied_axes=False, machines=mapping)
    for row in results:
        parent = row["meresi_pont"].rsplit("\\",1)[0]
        spec = mapping.get(parent)
        if not spec:
            raise ValueError(f"Adj gépazonosítót és típust az alkalmi_gepek.csv új pont_szulo sorához: {parent}")

    return results


def aggregate(group):
    values = {key: [] for key in COLUMNS}
    for r in group:
        if r["adathibak"] or r["allapot"] == "MINOSEGI_JELZES" or not r["frekvenciatengely"]:
            continue
        f = r["felhasznalt_jellemzok"]
        unit = r["mertekegyseg"].casefold()
        det = r["detektalas"].casefold()
        if (unit, det) == ("mm/s", "rms"):
            keys = dict(seb_1x="harm_1_amplitudo", seb_2x="harm_2_amplitudo",
                        seb_3x="harm_3_amplitudo", seb_2_per_1="harm_2_1_amplitudo_arany")
        elif (unit, det) == ("g", "peak"):
            keys = {"gyors_max": "spektrum_max_amplitudo"}
        elif (unit, det) == ("ge", "peak to peak") and r["meresi_pont"].upper().endswith("-E3"):
            keys = {"burk_E3_max": "spektrum_max_amplitudo"}
        else:
            continue
        for key, source in keys.items():
            if f.get(source) is not None:
                values[key].append((float(f[source]),r["meresi_pont"],r["meres_id"]))
    row = {}
    for key, candidates in values.items():
        value, point, record = max(candidates) if candidates else (None,"","")
        row.update({key:value, key+"_pont":point, key+"_rekord":record})
    return row


def inventory(results):
    rows = []
    for machine, group in sessions(results):
        types = {r["geptipus"] for r in group}
        if len(types)!=1:
            raise ValueError(f"Eltérő típusok ugyanahhoz a géphez: {machine}")
        ids = sorted(r["meres_id"] for r in group)
        ident = hashlib.sha256("|".join(ids).encode()).hexdigest()[:24]
        symptoms = sorted({s for r in group for s in r["lehetseges_hibairanyok"]})
        rows.append(dict(alkalom_id=ident, gep=machine, geptipus=types.pop(),
                         kezdet=group[0]["idopont"], veg=group[-1]["idopont"], spektrumok=len(group),
                         felteteles_skala=any(r["felteteles_frekvenciaskala"] for r in group),
                         skf_megfigyelesek=" / ".join(symptoms), rekordok="|".join(ids), **aggregate(group)))
    return rows


def feature_matrix(table):
    import numpy as np
    import pandas as pd
    return table[COLUMNS].apply(
        lambda column: pd.to_numeric(column.astype("string").str.replace(",", ".", regex=False)
                                     .replace("", pd.NA)).astype(float)
    ).replace([np.inf, -np.inf], np.nan)


def training(table, kind, out):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from mi_modell import felosztas, modellek, mutatok
    part = table[table.geptipus.eq(kind) & table.allapot.isin(["hibas","megfelelo"])].copy()
    if not table.allapot.isin(["ismeretlen", "hibas", "megfelelo"]).all():
        raise ValueError("Érvénytelen alkalomcímke.")
    if part.empty or part.allapot.nunique()!=2:
        raise ValueError("Mindkét állapotból igazolt alkalmak szükségesek a kiválasztott géptípushoz.")
    if part.bizonyitek.str.strip().eq("").any():
        raise ValueError("A besorolt alkalmakhoz add meg az állapot forrását.")
    if part.felteteles_skala.astype(str).str.lower().eq("true").any():
        raise ValueError("Tanítás előtt ellenőrizd a frekvenciaskálát és készíts új importot az SKF-skálabeállítással.")
    X = feature_matrix(part)
    if X.isna().all(axis=1).any():
        raise ValueError("Van használható jellemző nélküli alkalom.")
    y = part.allapot.eq("hibas").astype(int).to_numpy()
    dev, test, cv = felosztas(y, part.gep.to_numpy())
    ranks, scores, candidates = {}, {}, {}
    for name,(model,grid) in modellek().items():
        fit = GridSearchCV(model, grid, cv=cv, scoring="balanced_accuracy", error_score="raise")
        fit.fit(X.iloc[dev], y[dev])
        ranks[name], candidates[name] = fit.best_score_, fit.best_estimator_
    winner = max(ranks,key=ranks.get)
    report = part.iloc[test][["alkalom_id","gep","allapot"]].copy()
    for name,model in candidates.items():
        pred = model.predict(X.iloc[test])
        scores[name] = mutatok(y[test],pred)
        report[name] = np.where(pred,"hibas","megfelelo")
    report.to_csv(out/"teszt.csv", index=False, sep=";", encoding="utf-8-sig")
    joblib.dump(dict(formatum=FORMAT, geptipus=kind, columns=COLUMNS, modell=candidates[winner],
                     tanitogepek=sorted(set(part.iloc[dev].gep))),out/"alkalmi_modell.joblib")
    (out/"tanitas.json").write_text(json.dumps(dict(valasztott=winner,belso_cv=ranks,teszt=scores,
        tanito_alkalmak=part.iloc[dev].alkalom_id.tolist(),teszt_alkalmak=report.alkalom_id.tolist()),ensure_ascii=False,indent=2),encoding="utf8")


def discover(data):
    from skf_diagnosztika.skf_fajlkezeles import REQUIRED
    canonical = "MINTA CSAK HIBÁK ELŐTTI MÉRÉSEK(csv).csv"
    files = sorted(p for p in data.rglob("*") if p.is_file() and p.suffix.casefold() == ".csv")
    exact = [p for p in files if p.name.casefold() == canonical.casefold()]
    if exact:
        return exact + [p for p in files if p.name.casefold() in GOOD_EXPORTS]
    if files and all(p.name.casefold() in GOOD_EXPORTS for p in files):
        return files
    matches = []
    for path in files:
        with path.open("rb") as handle:
            prefix = handle.read(3)
        enc = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        with path.open(encoding=enc, errors="replace") as handle:
            line = next((line for line in handle if line.strip()), "")
        sep = max((";", ",", "\t"), key=line.count)
        header = [v.strip() for v in next(csv.reader([line], delimiter=sep))]
        if set(REQUIRED) <= set(header) and "phase" not in path.stem.casefold():
            matches.append(path)
    if len(matches) > 1:
        names = "\n".join(str(p) for p in matches)
        raise ValueError("Több spektrumexport található. Az import után add meg a kívánt fájlok útvonalát:\n" + names)
    if not matches:
        raise ValueError(f"Nincs SKF spektrum-CSV itt: {data.resolve()}. Másold ide vagy add meg az import parancs után.")
    return matches


@contextmanager
def import_zar(mappa):
    # Az operációs rendszer a program leállásakor is feloldja a fájlzárat.
    with (mappa / ".alkalmi_import.lock").open("a+b") as fajl:
        try:
            if os.name == "nt":
                import msvcrt
                fajl.seek(0)
                msvcrt.locking(fajl.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fajl, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError(f"Az importmappa nem zárolható: {mappa}. Próbáld újra az importot.") from exc
        yield


def egyesitett_alkalmak(path, rows):
    alkalmak = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as fajl:
            olvaso = csv.DictReader(fajl, delimiter=";")
            if not {"alkalom_id", "rekordok", "gep", *COLUMNS} <= set(olvaso.fieldnames or []):
                raise ValueError(f"Hiányos alkalomtábla: {path}")
            for sor in olvaso:
                azonosito = sor["alkalom_id"]
                if not azonosito or azonosito in alkalmak:
                    raise ValueError(f"Hiányzó vagy ismétlődő alkalomazonosító: {path}")
                alkalmak[azonosito] = sor

    for sor in rows:
        azonosito = sor["alkalom_id"]
        alkalmak[azonosito] = {**alkalmak.get(azonosito, {}), **sor}

    # Ugyanaz a spektrum csak egyetlen tanítási alkalomhoz tartozhat.
    latott = set()
    for sor in alkalmak.values():
        meresek = set(sor["rekordok"].split("|"))
        if latott.intersection(meresek):
            raise ValueError("Részben átfedő mérési alkalmak. Importáld együtt az érintett exportokat egy új mappába.")
        latott.update(meresek)
    return list(alkalmak.values())


def import_mentese(path, labels, rows, mind_hibas, bizonyitek):
    teljes = egyesitett_alkalmak(path, rows)
    old = {}
    if labels.exists():
        with labels.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if not {"alkalom_id", "allapot", "bizonyitek"} <= set(reader.fieldnames or []):
                raise ValueError(f"Hiányos címkefejléc: {labels}")
            for item in reader:
                key = item.pop("alkalom_id")
                if key in old:
                    raise ValueError(f"Ismételt alkalomcímke: {key}")
                old[key] = item
    for r in rows:
        key = r["alkalom_id"]
        if key in CONFIRMED_GOOD:
            if mind_hibas or old.get(key, {}).get("allapot") == "hibas":
                raise ValueError("A hibás címke ütközik az igazolt megfelelő állapottal: " + key)
            if old.get(key, {}).get("allapot") in (None, "ismeretlen"):
                old[key] = {k:r[k] for k in ("gep","geptipus","kezdet","spektrumok")}
                old[key].update(allapot="megfelelo", bizonyitek="Felhasználói igazolás: " + CONFIRMED_GOOD[key])
            continue
        if key not in old or (mind_hibas and old[key]["allapot"]=="ismeretlen"):
            old[key] = {k:r[k] for k in ("gep","geptipus","kezdet","spektrumok")}
            old[key].update(allapot="hibas" if mind_hibas else "ismeretlen", bizonyitek=bizonyitek)
        elif mind_hibas and old[key]["allapot"] != "hibas":
            raise ValueError("A megadott állapot ütközik egy korábbi alkalomcímkével.")

    # Előbb mindkét tábla elkészül. Új alkalom csak a címkéje mentése után jelenik meg.
    with tempfile.TemporaryDirectory(dir=path.parent) as atmeneti:
        cel = Path(atmeneti)
        write_csv(cel / path.name, teljes)
        write_csv(cel / labels.name, [dict(alkalom_id=k, **v) for k, v in old.items()])
        (cel / labels.name).replace(labels)
        (cel / path.name).replace(path)
    return teljes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("csv",type=Path,nargs="*",help="Egy vagy több eredeti spektrum-CSV")
    imp.add_argument("--gepek",type=Path,default=ROOT/"alkalmi_gepek.csv")
    imp.add_argument("--konfig",type=Path,default=ROOT/"skf_beallitasok.json")
    imp.add_argument("--pontok",type=Path,default=ROOT/"skf_pontbeallitasok.csv")
    imp.add_argument("--mind-hibas",action="store_true",help="Minden bemeneti alkalom igazoltan hibás")
    imp.add_argument("--bizonyitek",default="")
    train = sub.add_parser("tanitas")
    train.add_argument("--geptipus",required=True)
    predict = sub.add_parser("becsles")
    predict.add_argument("--modell",type=Path,required=True)
    for cmd in (imp,train,predict):
        cmd.add_argument("--mappa",type=Path,default=ROOT/"outputs"/"alkalmi_mi")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv or ["import"])
    out = args.mappa
    out = out.resolve()
    print(f"Kimeneti mappa: {out}", flush=True)
    path = out/"alkalmak.csv"
    labels = out/"alkalmi_cimkek.csv"
    try:
        if args.op == "import":
            if args.mind_hibas and not args.bizonyitek.strip():
                raise ValueError("A csoportos hibás címkézéshez add meg a --bizonyitek értékét.")
            paths = args.csv or discover(ROOT/"data")
            for source in paths:
                print(f"Feldolgozott spektrumfájl: {source.resolve()}", flush=True)
            rows = inventory(collect(paths,args.gepek,load_config(args.konfig),load_point_settings(args.pontok)))
            out.mkdir(parents=True, exist_ok=True)
            with import_zar(out):
                teljes = import_mentese(path, labels, rows, args.mind_hibas, args.bizonyitek)
            print(f"Címketábla: {labels}")
            print(f"Összesen {len(teljes)} mérési alkalom, {len({r['gep'] for r in teljes})} gép. {path}")
        elif args.op == "tanitas":
            table = read_csv(path).merge(read_csv(labels)[["alkalom_id","allapot","bizonyitek"]],
                on="alkalom_id",validate="one_to_one")
            run = Path(tempfile.mkdtemp(prefix="tanitas_",dir=out))
            training(table,args.geptipus,run)
            print(run)
        else:
            import joblib
            import numpy as np
            import pandas as pd
            model = joblib.load(args.modell)
            if model["formatum"] != FORMAT or model["columns"] != COLUMNS:
                raise ValueError("Eltérő alkalmi modellformátum.")
            table = read_csv(path)
            X = feature_matrix(table)
            valid = table.geptipus.eq(model["geptipus"]) & X.notna().any(axis=1) & table.felteteles_skala.str.lower().eq("false")
            table.insert(table.columns.get_loc("gep") + 1, "becsles", "nem_ertekelheto")
            if valid.any():
                table.loc[valid,"becsles"] = np.where(model["modell"].predict(X.loc[valid]),"hibas","megfelelo")
            if "skf_megfigyelesek" in table.columns:
                table.loc[table["becsles"] != "hibas", "skf_megfigyelesek"] = ""
            table.to_csv(out/"alkalmi_becslesek.csv",index=False,sep=";",encoding="utf-8-sig")
            print(out/"alkalmi_becslesek.csv")
        return 0
    except (ValueError, KeyError, OSError, ImportError) as exc:
        message = f"{args.op}: {exc}"
        if isinstance(exc, ImportError):
            message += "\nTelepítés: python -m pip install -r requirements_mi.txt"
        print("Hiba:", message)
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out/"utolso_hiba.txt").write_text(message + "\n", encoding="utf-8-sig")
        except OSError as log_error:
            print("A hibajelentés mentése sikertelen:", log_error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
