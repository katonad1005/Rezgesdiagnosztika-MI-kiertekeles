# A géphez és a mérési ponthoz tartozó beállításokat az SKF-útvonal alapján választja ki.

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .skf_fajlkezeles import decode, number, table, timestamp, write_csv

DEFAULTS = {
    "frekvenciaegyseg": None, "kezdo_frekvencia_hz": None,
    "vegpontok": "ket_vegpont", "skala_ellenorzott": False,
    "fordulat_igazolt": False, "fordulati_frekvencia_hz": None,
    "fordulat_forrasa": "", "meresi_irany": "auto", "gep": "",
    "lapatszam": None, "fogszam": None, "tengelykapcsolo": None, "villanymotor": None,
    "tapfrekvencia_hz": None, "polusszam": None,
    "csapagy_tipus": "", "csapagy_adatok_forrasa": "",
    "BPFO_order": None, "BPFI_order": None, "BSF_order": None, "FTF_order": None,
}
TECHNICAL = {"frekvencia_tures_szazalek": 1.5,
             "minimum_hatter_arany": 3.0, "minimum_relativ_amplitudo": 0.005}
LEGACY_SPEED_FIELDS = {'fordulat_igazolt', 'fordulati_frekvencia_hz', 'fordulat_forrasa'}
POINT_FIELDS = ["meresi_pont", "meresek_szama", "mertekegyseg", "export_fordulat_hz",
                "export_vegfrekvencia"] + list(DEFAULTS)


def fresh_config():
    return {"verzio": 1, "alap": copy.deepcopy(DEFAULTS), "pont_profilok": [],
            "csucskereses": dict(TECHNICAL)}


def save_config(path, config):
    Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def validate_values(values):
    if not isinstance(values, dict):
        raise ValueError("A beállításoknak JSON-objektumnak kell lenniük")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError("Ismeretlen beállítás: " + ", ".join(sorted(unknown)))
    for key, value in values.items():
        if key in LEGACY_SPEED_FIELDS:
            continue  # Korábbi beállítások beolvasásához megőrzött mező.
        if value is None or value == "":
            continue
        if key in {"frekvenciaegyseg", "vegpontok", "meresi_irany", "gep", "fordulat_forrasa", "csapagy_tipus", "csapagy_adatok_forrasa"} and not isinstance(value, str):
            raise ValueError(key + ": szöveges érték szükséges")
        if key in {"skala_ellenorzott", "fordulat_igazolt", "villanymotor", "tengelykapcsolo"}:
            if not isinstance(value, bool):
                raise ValueError(key + ": true vagy false szükséges")
        elif key == "frekvenciaegyseg":
            if value not in {"Hz", "CPM"}:
                raise ValueError("frekvenciaegyseg: Hz vagy CPM")
        elif key == "vegpontok":
            if value not in {"ket_vegpont", "felso_nelkul"}:
                raise ValueError("vegpontok: ket_vegpont vagy felso_nelkul")
        elif key == "meresi_irany":
            if value.casefold() not in {"auto", "ho", "ve", "hi", "hm", "ax", "radialis", "ismeretlen"}:
                raise ValueError("Nem támogatott mérési irány: " + value)
        elif key in {"lapatszam", "fogszam", "polusszam"}:
            x = number(value)
            if x != int(x) or x < 2 or (key == "polusszam" and int(x) % 2):
                raise ValueError(key + ": legalább 2 értékű egész szám szükséges (pólusszám: páros)")
        elif key.endswith("_hz") or key.endswith("_order"):
            x = number(value)
            if x < 0 or (key != "kezdo_frekvencia_hz" and x == 0):
                raise ValueError(key + ": érvénytelen szám")


def load_config(path):
    config = fresh_config()
    if not Path(path).exists():
        save_config(path, config)
        return config
    loaded = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("A konfiguráció gyökere JSON-objektum legyen")
    if set(loaded) - set(config) or loaded.get("verzio", 1) != 1:
        raise ValueError("Ismeretlen konfigurációs mező vagy verzió")
    validate_values(loaded.get("alap", {}))
    config["alap"].update(loaded.get("alap", {}))
    config["pont_profilok"] = loaded.get("pont_profilok", [])
    if not isinstance(config["pont_profilok"], list):
        raise ValueError("A pont_profilok mező listát vár")
    for profile in config["pont_profilok"]:
        if not isinstance(profile, dict):
            raise ValueError("Minden pontprofil JSON-objektum legyen")
        if set(profile) - {"nev", "aktiv", "prioritas", "utvonal_tartalmazza", "ertekek", "ervenyes_kezdete", "ervenyes_vege"}:
            raise ValueError("Ismeretlen profilmező")
        patterns = profile.get("utvonal_tartalmazza")
        if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) or not p.strip() for p in patterns):
            raise ValueError("A profilhoz legalább egy nem üres útvonalrészlet kell")
        if not isinstance(profile.get("aktiv", True), bool) or not isinstance(profile.get("prioritas", 0), int):
            raise ValueError("Hibás profilprioritás vagy aktiv jelző")
        validate_values(profile.get("ertekek", {}))
        start = timestamp(profile["ervenyes_kezdete"]) if profile.get("ervenyes_kezdete") else ""
        end = timestamp(profile["ervenyes_vege"]) if profile.get("ervenyes_vege") else ""
        if start and end and end < start:
            raise ValueError("A profil időintervalluma fordított")
    tech = loaded.get("csucskereses", {})
    if set(tech) - set(TECHNICAL):
        raise ValueError("Ismeretlen csúcskeresési paraméter")
    config["csucskereses"].update(tech)
    t = config["csucskereses"]
    if not (0 < number(t["frekvencia_tures_szazalek"]) <= 10 and
            number(t["minimum_hatter_arany"]) > 1 and
            0 < number(t["minimum_relativ_amplitudo"]) <= 1):
        raise ValueError("Érvénytelen technikai csúcskeresési paraméter")
    config["csucskereses"] = {key: number(value) for key, value in t.items()}
    return config


def load_point_settings(path):
    if not Path(path).exists():
        return {}
    rows = table(decode(path)[0])
    if not rows or "meresi_pont" not in rows[0]:
        raise ValueError("A pontbeállításból hiányzik a meresi_pont oszlop")
    if set(rows[0]) - set(POINT_FIELDS):
        raise ValueError("Ismeretlen oszlop a pontbeállításban")
    result = {}
    for line, cells in enumerate(rows[1:], 2):
        row = dict(zip(rows[0], cells))
        point = row.pop("meresi_pont", "").strip()
        if not point:
            if any(str(value).strip() for value in row.values()):
                raise ValueError(f"Pontbeállítás {line}. sor: hiányzó mérési pont")
            continue
        if point in result:
            raise ValueError("Ismétlődő mérési pont a beállításokban: " + point)
        values = {key: str(value).strip() for key, value in row.items()
                  if key in DEFAULTS and str(value).strip()}
        for key in ("skala_ellenorzott", "villanymotor", "tengelykapcsolo"):
            if key in values:
                lower = values[key].casefold()
                if lower not in {"igen", "nem", "true", "false", "1", "0"}:
                    raise ValueError(key + ": igen vagy nem szükséges")
                values[key] = lower in {"igen", "true", "1"}
        validate_values(values)
        result[point] = values
    return result


def resolve(point, config, point_settings, date=None):
    values = dict(config["alap"])
    sources = {key: "alap" for key in values}
    applied, errors, writes = [], [], {}
    profiles = sorted(config["pont_profilok"], key=lambda p: p.get("prioritas", 0))
    for profile in profiles:
        if not profile.get("aktiv", True) or not all(p.casefold() in point.casefold()
                                                     for p in profile["utvonal_tartalmazza"]):
            continue
        start = timestamp(profile["ervenyes_kezdete"]) if profile.get("ervenyes_kezdete") else ""
        end = timestamp(profile["ervenyes_vege"]) if profile.get("ervenyes_vege") else ""
        if (start or end) and (not date or (start and date < start) or (end and date > end)):
            continue
        name = profile.get("nev", "névtelen profil")
        priority = profile.get("prioritas", 0)
        for key, value in profile.get("ertekek", {}).items():
            ident = (priority, key)
            if key not in LEGACY_SPEED_FIELDS and ident in writes and writes[ident] != value:
                errors.append("Azonos prioritású profilok ütköznek: " + key)
            writes[ident] = value
            values[key], sources[key] = value, name
        applied.append(name)
    for key, value in point_settings.get(point, {}).items():
        values[key], sources[key] = value, "skf_pontbeallitasok.csv"
    direction = values["meresi_irany"].casefold()
    if direction == "auto":
        leaf = re.split(r"[\\/]", point)[-1].upper()
        tokens = re.split(r"[^A-Z]+", leaf)
        found = sorted(set(tokens) & {"HO", "VE", "HI", "HM", "AX"})
        direction = found[0].casefold() if len(found) == 1 else "ismeretlen"
        sources["meresi_irany"] = "elnevezésből felismert" if len(found) == 1 else "ismeretlen"
    values["meresi_irany"] = direction
    return values, sources, applied, list(dict.fromkeys(errors))


def point_template(path, records):
    existing = table(decode(path)[0]) if Path(path).exists() else []
    old_rows = [dict(zip(existing[0], row)) for row in existing[1:]] if existing else []
    known = {row.get("meresi_pont", "") for row in old_rows}
    groups = {}
    for record in records:
        groups.setdefault(record.point, []).append(record)
    rows = []
    for point, group in sorted(groups.items()):
        if point in known:
            continue
        rows.append({"meresi_pont": point, "meresek_szama": len(group),
                     "mertekegyseg": " / ".join(sorted({r.meta["Unit"] for r in group})),
                     "export_fordulat_hz": " / ".join(sorted({r.meta["Speed (Hz)"] for r in group})),
                     "export_vegfrekvencia": " / ".join(sorted({r.meta["End Frequency"] for r in group}))})
    if rows or not Path(path).exists():
        write_csv(path, old_rows + rows, POINT_FIELDS)


def wizard(config):
    print("\nKözös beállítások; eltérő pontok később a skf_pontbeallitasok.csv-ben felülírhatók.")
    print("Csak a mérési beállításokból ellenőrzött értéket add meg. Üres válasz: ismeretlen.")
    unit = input("Az End Frequency egysége: 1 = Hz, 2 = CPM, Enter = ismeretlen: ").strip()
    start = input("Kezdőfrekvencia Hz-ben (pl. 0; Enter = ismeretlen): ").strip()
    ends = input("Pontkiosztás: 1 = mindkét végpont benne van, 2 = felső végpont nincs, Enter = ismeretlen: ").strip()
    print('A fordulatot automatikusan az 5–100 Hz közötti sávmaximum adja; a Speed mezőt nem használjuk.')
    config["alap"].update(frekvenciaegyseg={"1": "Hz", "2": "CPM"}.get(unit),
                          kezdo_frekvencia_hz=number(start) if start else None,
                          vegpontok="felso_nelkul" if ends == "2" else "ket_vegpont",
                          skala_ellenorzott=unit in {"1", "2"} and bool(start) and ends in {"1", "2"},
                          fordulat_igazolt=False, fordulati_frekvencia_hz=None, fordulat_forrasa='')
    validate_values(config["alap"])
    return config


def main_scale_overrides(meta, config_path, points_path):
    """Meglévő SKF-skálák átadása a főprogram előfeldolgozásához, kérdés nélkül."""
    config, point_settings = load_config(config_path), load_point_settings(points_path)
    rows = meta.to_dict("records") if hasattr(meta, "to_dict") else list(meta)
    overrides = {}
    for row in rows:
        values, _, _, errors = resolve(str(row["meresi_pont"]), config, point_settings, str(row["idopont"]))
        if errors:
            continue
        unit, start = values.get("frekvenciaegyseg"), values.get("kezdo_frekvencia_hz")
        if unit in {"Hz", "CPM"} and start not in (None, ""):
            endpoints = values.get("vegpontok", "ket_vegpont")
            origin = "atvett_skf_skala:" + endpoints
            if not values.get("skala_ellenorzott"):
                origin += ":ellenorizetlen"
            overrides[int(row["csv_sor"])] = {"egyseg": unit, "kezdo": number(start) * (60 if unit == "CPM" else 1),
                                             "forras": origin, "vegpontok": endpoints}
    return overrides
