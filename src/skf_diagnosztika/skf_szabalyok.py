# A spektrum jellemzői alapján megvizsgálja, mely hibákra utalhat a mérés.

from __future__ import annotations

import re
from statistics import median

from .skf_jellemzok import (Peaks, distinct_hits, is_hit, make_axis,
                          modality)
from .skf_fajlkezeles import number

VERSION = "1.4.0"
CATALOG = [
    ("ski_slope", "Ski-Slope – mérésminőség"),
    ("excentricitas", "Forgórész mechanikai excentricitása"),
    ("nem_szinkron_osszetevok", "Nem forgási harmonikusú spektrumösszetevők"),
    ("kiegyensulyozatlansag", "Kiegyensúlyozatlanság részleges mintázata"),
    ("beallitas_vagy_gorbult_tengely", "Tengelykapcsoló beállítási hibájával összeegyeztethető mintázat"),
    ("csapagy_BPFO", "Csapágy: külső gyűrű"),
    ("csapagy_BPFI", "Csapágy: belső gyűrű"),
    ("csapagy_BSF", "Csapágy: gördülőelem (2×BSF)"),
    ("csapagy_FTF", "Csapágy: kosár"),
    ("rotor_oldalsavok", "Pólusáthaladási oldalsávok"),
]
CANDIDATE_STATUSES = {"TUNET_TALALHATO", "RESZLEGES_TUNET"}


def outcome(code, name, status, text, peaks=None, missing=None, data=None):
    return {"szabaly": code, "megnevezes": name, "allapot": status, "indoklas": text,
            "hianyzo_vagy_tovabbi_adat": missing or [], "csucsok": peaks or [],
            "szamitott_adatok": data or {}}


def inspect_rule(code, name, record, settings, peaks, speed, mode):
    def result(status, text, hits=None, missing=None, data=None):
        return outcome(code, name, status, text, hits, missing, data)

    if code == "ski_slope":
        if mode != "sebesseg":
            return result("NEM_ALKALMAZHATO", "A megvalósított alakvizsgálat sebességspektrumra vonatkozik.")
        bands = [[], [], []]
        if peaks.axis.start <= 1 and peaks.axis.end >= 20:
            for i, value in enumerate(record.values):
                hz = peaks.axis.frequency(i)
                if hz > 20:
                    break
                if hz > 0:
                    bands[0 if hz <= 5 else 1 if hz <= 10 else 2].append(value)
        if min(map(len, bands)) < 3:
            return result("KORLATOZOTT_VIZSGALAT", "A 0–20 Hz tartomány lefedése vagy felbontása kevés az alakvizsgálathoz.")
        a, b, c = map(median, bands)
        descending = a > b > c and a >= 2*b and b >= 2*c
        return result("MINOSEGI_JELZES" if descending else "NINCS_AZONOSITOTT_TUNET",
                      "Három kisfrekvenciás sáv mediánja. Saját szűrőfeltétel: mindkét lépésben legalább feleződés.",
                      data={"savmedianok": [a, b, c]})
    if code == "kiegyensulyozatlansag" and settings["meresi_irany"] == "ax":
        return result("NEM_ALKALMAZHATO", "A mérés axiális. A radiális pontokat az alkalmi összesítés használja.")
    if code == "rotor_oldalsavok" and settings.get("villanymotor") is False:
        return result("NEM_ALKALMAZHATO", "A mérési pont a hajtott géphez tartozik.")
    if code == "beallitas_vagy_gorbult_tengely" and settings.get("tengelykapcsolo") is False:
        return result("NEM_ALKALMAZHATO", "A gépprofil szerint nincs tengelykapcsoló.")
    if speed is None:
        raise ValueError("Nincs a rendelkezésre álló spektrumból becsülhető tengelyfordulat")

    if code == "nem_szinkron_osszetevok":
        if mode not in {"gyorsulas", "burkologorbe"}:
            return result("NEM_ALKALMAZHATO", "Ez az áttekintés gyorsulás- és burkológörbe-spektrumhoz tartozik")
        candidates = []
        for key, value in peaks.features.items():
            if not re.fullmatch(r"csucs_\d+_index", key) or value is None:
                continue
            index = int(value)
            if not 1 <= index < len(record.values) - 1:
                continue
            frequency = peaks.axis.frequency(index)
            if frequency <= 0:
                continue
            order = frequency / speed
            if abs(frequency - round(order) * speed) <= peaks.tolerance(frequency):
                continue
            hit = peaks.find(frequency, speed)
            if is_hit(hit):
                candidates.append(dict(hit, fordulati_rend=order))
        found = distinct_hits(candidates)
        pairs = []
        for a in found:
            for b in found:
                if b["csucs_hz"] <= a["csucs_hz"]:
                    continue
                harmonic = round(b["csucs_hz"] / a["csucs_hz"])
                if harmonic in (2, 3) and abs(b["csucs_hz"] - harmonic * a["csucs_hz"]) <= peaks.tolerance(b["csucs_hz"]):
                    pairs.append({"alap_hz": a["csucs_hz"], "harmonikus_hz": b["csucs_hz"], "szorzo": harmonic})
        status = ("RESZLEGES_TUNET" if mode == "burkologorbe" and pairs else
                  "OSSZETEVO_TALALHATO" if found else "NINCS_AZONOSITOTT_TUNET")
        value = result(status,
                      "A nem egész fordulati rendű csúcsokat külön felsorolja. Ezek csapágy- és más gerjesztésből is származhatnak; hibatípust nem igazolnak.",
                      found, ["A konkrét csapágykomponens megnevezéséhez csapágytípus és hibafrekvencia-szorzók szükségesek"],
                      {"nem_szinkron_csucsok": len(found), "harmonikus_parok": pairs,
                       "eljárás": "Nem egész fordulati rendű csúcsok és harmonikus párok keresése"})
        if status == "RESZLEGES_TUNET":
            value["megnevezes"] = "Nem szinkron harmonikussor – csapágyeredet is lehetséges"
        return value

    if code.startswith("csapagy_"):
        if mode not in {"burkologorbe", "gyorsulas"}:
            raise ValueError("Gyorsulás- vagy burkológörbe-spektrum szükséges ehhez a csapágyvizsgálathoz")
        component = code.split("_")[1]
        order = settings.get(component + "_order")
        if order in (None, "") or not settings.get("csapagy_adatok_forrasa") or not settings.get("csapagy_tipus"):
            raise ValueError(component + " szorzó, csapágyazonosító és a hibafrekvenciák forrása szükséges")
        base = number(order) * speed * (2 if component == "BSF" else 1)
        hits = [dict(peaks.find(k * base, speed), harmonikus=k) for k in (1, 2, 3)]
        found = distinct_hits(hits)
        collisions = [p for p in found if abs(p["cel_hz"] - round(p["cel_hz"] / speed) * speed)
                      <= p["tures_hz"] + peaks.tolerance(round(p["cel_hz"] / speed) * speed)]
        missing = ["Időjel, trend és az adott csapágy beépítésének ellenőrzése"]
        if mode != "burkologorbe":
            missing.append("Korai csapágyhiba vizsgálatához burkológörbe-mérés is szükséges lehet")
        if collisions:
            missing.append("Forgási harmonikussal átfedő csapágycél; az ok nem különíthető el")
        status = "TUNET_TALALHATO" if len(found) >= 2 and not collisions else "RESZLEGES_TUNET" if found else "NINCS_AZONOSITOTT_TUNET"
        return result(status, "Csapágyfrekvencia és harmonikusainak illesztése; a találat lehetséges hibairány.",
                      hits, missing, {"alap_hibafrekvencia_hz": base, "onallo_talalatok": len(found),
                                      "forgasi_harmonikussal_atfed": bool(collisions)})

    if mode not in {"sebesseg", "gyorsulas", "elmozdulas"}:
        raise ValueError("Ehhez közvetlen rezgéssebesség-, gyorsulás- vagy elmozdulásspektrum szükséges")

    if code == "excentricitas":
        one = peaks.find(speed, speed)
        return result("NEM_KULONITHETO_EL", "Az 1× összetevő mechanikai excentricitásra nem fajlagos. Elkülönítéshez ütésmérés szükséges.",
                      [one], data={"1x_csucs": is_hit(one)})
    if code == "kiegyensulyozatlansag":
        if settings["meresi_irany"] not in {"ho", "ve", "hi", "hm", "radialis"}:
            raise ValueError("Radiális mérési irány szükséges")
        one = peaks.find(speed, speed)
        blade = int(number(settings["lapatszam"])) if settings.get("lapatszam") else None
        orders = [k for k in range(2, 11) if not blade or k % blade]
        hits = [one] + [dict(peaks.find(k * speed, speed), rend=k) for k in orders]
        amplitude = one.get("amplitudo", 0)
        ratios = {str(k): (peaks.band_max(k * speed) / amplitude
                          if amplitude and peaks.band_max(k * speed) is not None else None) for k in orders}
        complete = all(value is not None for value in ratios.values()) and all(
            p["allapot"] != "NEM_FELBONTHATO" for p in hits)
        supports = is_hit(one) and complete and all(value < 0.15 for value in ratios.values())
        partial = is_hit(one) and not complete
        return result("RESZLEGES_TUNET" if supports else "KORLATOZOTT_VIZSGALAT" if partial else "NINCS_AZONOSITOTT_TUNET",
                      "A program azt vizsgálja, hogy a figyelembe vett 2×–10× harmonikusok amplitúdója egyenként kisebb-e a radiális 1× amplitúdó 15%-ánál.",
                      hits, ["Normálszint vagy trend; fázismérés. A normálhoz képest magas 1× amplitúdó nincs igazolva."] +
                      (["A lapátszám hiányzik; a lapátáthaladás nem lett külön kezelve"] if blade is None else []) +
                      (["A 2×–10× vizsgálati tartomány hiányos"] if not complete else []),
                      {"harmonikus_aranyok": ratios, "kihagyott_lapatharmonikusok": [k for k in range(2, 11) if k not in orders],
                       "kis_harmonikus_mintazat": supports, "reszvizsgalat": "1×–10×; a teljes spektrumot nem helyettesíti"})

    if code == "beallitas_vagy_gorbult_tengely":
        hits = [peaks.find(k * speed, speed) for k in (1, 2)]
        a, b = (p.get("amplitudo", 0) for p in hits)
        ratio = b / a if a > 0 else None
        supports = all(is_hit(p) for p in hits) and ratio >= 0.30
        return result("RESZLEGES_TUNET" if supports else "NINCS_AZONOSITOTT_TUNET",
                      "Mindkét csúcs jelenléte esetén a feltétel az, hogy a 2× amplitúdó elérje az 1× amplitúdó 30%-át. A két lehetséges ok hasonló spektrumot adhat.",
                      hits, ["Gép- és tengelykapcsoló-elrendezés, normálszint, radiális/axiális fázismérés"],
                      {"2x_per_1x": ratio, "elkulonitett_hibatipus": None})

    if settings.get("villanymotor") is not True:
        raise ValueError("A vizsgált tengely villanymotorhoz tartozását meg kell adni")
    if not settings.get("tapfrekvencia_hz"):
        raise ValueError("A motor méréskori tényleges tápfrekvenciája szükséges; frekvenciaváltónál nem feltétlenül 50 Hz")
    supply = number(settings["tapfrekvencia_hz"])
    if not settings.get("polusszam"):
        raise ValueError("A pólusáthaladáshoz a motor pólusszáma szükséges")
    poles = number(settings["polusszam"])
    synchronous = 2 * supply / poles
    spacing = poles * (synchronous - speed)
    if spacing <= 0 or spacing >= speed:
        raise ValueError("A megadott fordulat, pólusszám és tápfrekvencia nem alkalmas erre az indukciósmotor-vizsgálatra")
    hits, pairs = [], 0
    for k in range(1, 6):
        carrier = dict(peaks.find(k * speed, min(speed, spacing)), rend=k, oldal=0)
        pair = [dict(peaks.find(k * speed + sign * spacing, spacing), rend=k, oldal=sign) for sign in (-1, 1)]
        hits.extend([carrier] + pair)
        pairs += int(is_hit(carrier) and all(is_hit(p) for p in pair))
    unresolved = any(p["allapot"] == "NEM_FELBONTHATO" for p in hits)
    return result("HIANYZO_ADAT" if unresolved else "RESZLEGES_TUNET" if pairs else "NINCS_AZONOSITOTT_TUNET",
                  "Pólusáthaladási távolságú oldalsávok az 1×–5× forgási összetevők körül.",
                  hits, ["Áramanalízis vagy további villamos vizsgálat"] +
                  (["Az oldalsávokhoz finomabb frekvenciafelbontás / szűkebb technikai keresősáv kell"] if unresolved else []),
                  {"szinkron_fordulat_hz": synchronous, "polusathaladas_hz": spacing, "oldalsavparok": pairs})


def evaluate(record, settings, technical, settings_errors=(), known_faulty=False, context=None):
    if context is None:
        # Közvetlen híváskor is a közös fordulatbecslés fut.
        # Az ellenőrzött skálát a megadott mérési beállítások határozzák meg.
        from .skf_elokeszites import automatic_context
        supplied_axis, issue = None, ''
        try:
            supplied_axis = make_axis(record, settings)
        except ValueError as exc:
            issue = str(exc)
        context = automatic_context(record, settings, technical, axis=supplied_axis,
                                    axis_supplied=True, axis_origin='megadott_meresi_skala')
        context['axis_issue'] = issue
    errors = list(record.errors) + list(settings_errors)
    axis, peaks, speed, speed_origin = None, None, None, ""
    axis_error, speed_error = "", ""
    if not errors:
        axis, axis_error = context.get("axis"), context.get("axis_issue", "")
        speed, speed_origin = context.get("speed"), context.get("speed_origin", "")
        speed_error = context.get("speed_issue", "")
        if axis:
            peaks = Peaks(record, axis, technical, context.get("features"))
    mode = modality(record.meta.get("Unit", ""))
    rules = []
    for code, name in CATALOG:
        if errors or axis is None:
            rules.append(outcome(code, name, "HIBAS_ADAT" if errors else "HIANYZO_ADAT",
                                 "; ".join(errors) if errors else axis_error))
            continue
        if code.startswith("csapagy_") and not (settings.get("csapagy_tipus") and settings.get("csapagy_adatok_forrasa") and settings.get(code.split("_")[1] + "_order")):
            continue
        if code.startswith("csapagy_") and mode not in {"burkologorbe", "gyorsulas"}:
            rules.append(outcome(code, name, "NEM_ALKALMAZHATO", "Ez a csapágy-részvizsgálat gyorsulás/burkológörbe mérést használ"))
            continue
        if code in {"kiegyensulyozatlansag", "beallitas_vagy_gorbult_tengely", "excentricitas", "rotor_oldalsavok"} and mode == "burkologorbe":
            rules.append(outcome(code, name, "NEM_ALKALMAZHATO", "Ehhez a vizsgálathoz közvetlen rezgésspektrum szükséges, nem burkológörbe"))
            continue
        try:
            rule = inspect_rule(code, name, record, settings, peaks, speed, mode)
            if rule["allapot"] == "NINCS_AZONOSITOTT_TUNET" and rule["csucsok"] and all(
                    p["allapot"] in {"TARTOMANYON_KIVUL", "NEM_FELBONTHATO"} for p in rule["csucsok"]):
                rule["allapot"] = "HIANYZO_ADAT"
                rule["hianyzo_vagy_tovabbi_adat"].append("A célfrekvenciák nem vizsgálhatók a rendelkezésre álló tartományon/felbontással")
            if context and (context.get("axis_conditional") or context.get("speed_conditional")) and rule["allapot"] == "TUNET_TALALHATO":
                rule["allapot"] = "RESZLEGES_TUNET"
                rule["hianyzo_vagy_tovabbi_adat"].append("A frekvenciaskála feltételes vagy a fordulat a vállalati 5–100 Hz sávmaximumából becsült")
            if rule["allapot"] == "NINCS_AZONOSITOTT_TUNET" and any(p["allapot"] in {"TARTOMANYON_KIVUL", "NEM_FELBONTHATO"} for p in rule["csucsok"]):
                rule["allapot"] = "KORLATOZOTT_VIZSGALAT"
            rules.append(rule)
        except ValueError as exc:
            rules.append(outcome(code, name, "HIANYZO_ADAT", str(exc), missing=[str(exc)]))
    candidates = [r["megnevezes"] for r in rules if r["allapot"] in CANDIDATE_STATUSES]
    if any(r["allapot"] == "MINOSEGI_JELZES" for r in rules):
        candidates = []
    evaluable = sum(r["allapot"] not in {"HIANYZO_ADAT", "HIBAS_ADAT", "TOVABBI_MERES_SZUKSEGES", "NEM_ALKALMAZHATO", "LEIRO_KIERTEKELES"} for r in rules)
    missing = [r["megnevezes"] for r in rules if r["allapot"] in {"HIANYZO_ADAT", "TOVABBI_MERES_SZUKSEGES"}]
    if mode in {"gyorsulas", "burkologorbe"} and not any(r["szabaly"].startswith("csapagy_") for r in rules):
        missing.append("Csapágykomponens azonosítása: típus, forrás és hibafrekvencia-szorzó")
    status = ("HIBAS_ADAT" if errors else "MINOSEGI_JELZES" if any(r["allapot"] == "MINOSEGI_JELZES" for r in rules) else "TUNET_TALALHATO" if candidates else
              "RESZLEGES_KIERTEKELES" if context is not None and evaluable and missing else
              "NINCS_AZONOSITOTT_TUNET" if evaluable else "CSAK_LEIRO_ADATOK" if context is not None else "NEM_ERTEKELHETO")
    parts = [x for x in re.split(r"[\\/]", record.point) if x]
    parent = "\\".join(parts[:-1]) if len(parts) > 1 else record.point
    return {"meres_id": record.id, "rekord_id": record.id, "meresi_pont": record.point, "idopont": record.date or record.meta.get("DTS"),
            "gepcsoport": settings.get("gep") or parent,
            "gepcsoport_eredete": "megadott" if settings.get("gep") else "útvonal szülője; ellenőrizendő csoportosítás",
            "forrasok": record.sources, "mertekegyseg": record.meta.get("Unit"),
            "detektalas": record.meta.get("Detection"), "meresmod": mode,
            "meresi_irany": settings["meresi_irany"], "pontszam": len(record.values),
            "allapot": status, "ismert_gepallapot": "HIBAS_FELHASZNALOI_KOZLES" if known_faulty else "ISMERETLEN",
            "allapot_eredete": "--ismerten-hibas kapcsoló" if known_faulty else "nincs megadva",
            "lehetseges_hibairanyok": candidates, "ertekelheto_szabalyok": evaluable,
            "adatot_igenylo_vizsgalatok": missing,
            "felhasznalt_jellemzok": context.get("features", {}) if context else {},
            "jellemzok_eredete": context.get("feature_origin", "") if context else "sajat_reszvizsgalat",
            "frekvenciatengely_eredete": context.get("axis_origin", "") if context else "kezileg_ellenorzott_beallitas",
            "felteteles_frekvenciaskala": bool(context and context.get("axis_conditional")),
            "felteteles_fordulat": bool(context and context.get("speed_conditional")),
            "adatfelhasznalasi_megjegyzesek": context.get("warnings", []) if context else [],
            "ujrafelhasznalt_harmonikus_savmaximumok": sorted(peaks.reused) if peaks else [],
            "fordulati_frekvencia_hz": speed, "fordulatszam_perc": speed * 60 if speed else None,
            "fordulat_eredete": speed_origin, "hianyzo_fordulat_indoka": speed_error,
            "fordulat_becsles": context.get('speed_estimate', {}) if context else {},
            "export_speed_nyers": record.meta.get('Speed (Hz)', ''),
            "frekvenciatengely": {"kezdet_hz": axis.start, "utolso_pont_hz": axis.end, "pontkoz_hz": axis.step} if axis else None,
            "hianyzo_skala_indoka": axis_error, "adathibak": errors,
            "maximum_amplitudo": max(record.values, default=None) if not record.errors else None,
            "beallitasok": settings, "szabalyok": rules,
            "megjegyzes": "A tünet hiánya nem igazol megfelelő állapotot. A jelzett okok nem automatikus tanítási címkék."}
