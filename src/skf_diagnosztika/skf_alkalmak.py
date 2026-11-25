# Gépenként mérési alkalmakba rendezi a spektrumokat, és összesíti a nyolc vizsgált eset jelzéseit.

import re
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

CASES = (
    ("Dynamic Imbalance", ("kiegyensulyozatlansag",)),
    ("Static Imbalance", ("kiegyensulyozatlansag",)),
    ("Couple Imbalance", ("kiegyensulyozatlansag",)),
    ("Rolling Element Bearing Wear", ("nem_szinkron_osszetevok", "csapagy_")),
    ("Eccentric Rotor", ("excentricitas",)),
    ("Cracked or Broken Rotor Bars", ("rotor_oldalsavok",)),
    ("Coupling Faults", ("beallitas_vagy_gorbult_tengely",)),
    ("Ski-Slope", ("ski_slope",)),
)
POSITIVE = {"TUNET_TALALHATO", "RESZLEGES_TUNET", "MINOSEGI_JELZES"}
UNKNOWN = {"HIANYZO_ADAT", "KORLATOZOTT_VIZSGALAT", "HIBAS_ADAT"}


def machine_key(row):
    if row.get("fizikai_gep"):
        return row["fizikai_gep"]
    parts = row["meresi_pont"].strip("\\").split("\\")
    parent = parts[:-1]
    configured = row["beallitasok"].get("gep")
    if parent and parent[-1].upper().endswith(("_MOTOR", "_TENGELY")):
        return configured or ("\\" + "\\".join(parent[:-1]) if parent[:-1] else row["meresi_pont"])
    # A légkezelő külön hajtásai külön egységként szerepelnek.
    if any(p.upper().endswith(("_BEFUVO", "_ELSZIVO")) for p in parent):
        return "\\".join(parent)
    return configured or row["meresi_pont"]


def sessions(results):
    machines = defaultdict(list)
    for row in results:
        machines[machine_key(row)].append(row)
    for machine, rows in sorted(machines.items()):
        group, seen, start = [], set(), None
        for row in sorted(rows, key=lambda r: (str(r["idopont"]), r["meres_id"])):
            try:
                when = datetime.fromisoformat(str(row["idopont"]))
            except ValueError:
                yield machine, [row]
                continue
            channel = (row["meresi_pont"], row["mertekegyseg"], row["detektalas"], row.get("csatorna"))
            if group and (when.date() != start.date() or when - start > timedelta(minutes=30) or channel in seen):
                yield machine, group
                group, seen = [], set()
            if not group:
                start = when
            group.append(row)
            seen.add(channel)
        if group:
            yield machine, group


def coupling_pair(evidence):
    for (a, _), (b, _) in combinations(evidence, 2):
        def end(row):
            tags = re.split("[^A-Z]+", row["meresi_pont"].split("\\")[-1].upper())
            return set(tags) & {"MD", "TD"}
        if {frozenset(end(a)), frozenset(end(b))} != {frozenset({"MD"}), frozenset({"TD"})}:
            continue
        if a["meresi_irany"] == "ismeretlen" or any(a[k] != b[k] for k in ("meresi_irany", "mertekegyseg", "detektalas")):
            continue
        resolution = max(r["frekvenciatengely"]["pontkoz_hz"] for r in (a, b))
        if abs(a["fordulati_frekvencia_hz"] - b["fordulati_frekvencia_hz"]) <= resolution:
            return a, b
    return None


def summarize(case, evidence):
    usable = [(r, s) for r, s in evidence if case == "Ski-Slope" or r["allapot"] != "MINOSEGI_JELZES"]
    positives = [(r, s) for r, s in usable if s["allapot"] in POSITIVE]
    if positives:
        if case.endswith("Imbalance"):
            return "ALTIPUS_NEM_KULONITHETO_EL", "Radiális kiegyensúlyozatlansági mintázat. A három altípus elkülönítéséhez fázisadat szükséges.", positives
        if case == "Coupling Faults":
            pair = coupling_pair(positives)
            if pair:
                return "RESZLEGES_TUNET", "Az MD és TD ponton azonos irányban, összeillő fordulatnál egyaránt látható az 1×–2× mintázat.", [(r,s) for r,s in positives if r in pair]
            return "RESZLEGES_TUNET", "1×–2× mintázat látható. Összeillő kétoldali MD–TD találat nincs.", positives
        if case == "Ski-Slope":
            return "MINOSEGI_JELZES", "Lejtős kisfrekvenciás spektrumkép. E spektrum géphibatünetei kimaradnak az alkalmi következtetésből.", positives
        return "RESZLEGES_TUNET", positives[0][1]["indoklas"], positives
    checked = [(r,s) for r,s in usable if s["allapot"] != "NEM_ALKALMAZHATO"]
    if not checked:
        if usable:
            return "NEM_ALKALMAZHATO", "Az alkalom mérési pontjaira vagy mérésmódjaira ez a részvizsgálat nem alkalmazható.", []
        return "KORLATOZOTT_VIZSGALAT", "Nincs felhasználható spektrum ehhez az esethez.", []
    if all(s["allapot"] == "HIBAS_ADAT" for _,s in checked):
        return "HIBAS_ADAT", checked[0][1]["indoklas"], checked
    if all(s["allapot"] == "HIANYZO_ADAT" for _,s in checked):
        return "HIANYZO_ADAT", " / ".join(dict.fromkeys(s["indoklas"] for _,s in checked)), checked
    if case == "Rolling Element Bearing Wear" and not any(s["szabaly"].startswith("csapagy_") for _,s in checked):
        return "KORLATOZOTT_VIZSGALAT", "Nincs értékelhető komponensvizsgálat. Csapágytípus, forrás és hibafrekvencia-szorzó szükséges; az általános csúcsvizsgálat külön látható.", checked
    if any(s["allapot"] == "NEM_KULONITHETO_EL" for _,s in checked):
        return "NEM_KULONITHETO_EL", "A mechanikai excentricitás az 1× összetevő alapján nem különíthető el.", checked
    completed = [(r,s) for r,s in checked if s["allapot"] == "NINCS_AZONOSITOTT_TUNET"]
    limited = [(r,s) for r,s in checked if s["allapot"] in UNKNOWN]
    if completed:
        return "NINCS_AZONOSITOTT_TUNET", "Az értékelhető részvizsgálatokban a keresett mintázat nem látszik." + (" További pontok vizsgálata korlátozott." if limited else ""), completed
    reasons = list(dict.fromkeys(s["indoklas"] for _,s in checked))
    return "KORLATOZOTT_VIZSGALAT", " / ".join(reasons), checked


def session_rows(results):
    output = []
    for machine, group in sessions(results):
        for case, codes in CASES:
            evidence = [(r,s) for r in group for s in r["szabalyok"]
                        if s["szabaly"] in codes or ("csapagy_" in codes and s["szabaly"].startswith("csapagy_"))]
            status, reason, used = summarize(case, evidence)
            output.append(dict(gep=machine, kezdet=group[0]["idopont"], veg=group[-1]["idopont"],
                               spektrumok=len(group), eset=case, allapot=status, indoklas=reason,
                               felhasznalt_meresek=list(dict.fromkeys(r["meres_id"] for r,s in used)),
                               alkalom_meresei=[r["meres_id"] for r in group]))
    return output
