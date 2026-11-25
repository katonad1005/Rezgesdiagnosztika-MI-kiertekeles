# A mérési alkalmak, a spektrumok és az egyes szabályok eredményeit CSV-táblákba írja.

from collections import defaultdict
from pathlib import Path

from .skf_alkalmak import session_rows
from .skf_fajlkezeles import write_csv


def write_reports(destination, records, results, inventory, manifest):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summaries, rule_rows, peak_rows, trend_rows = [], [], [], []
    daily = defaultdict(list)
    for record, result in zip(records, results):
        row = {key: result[key] for key in ("meres_id", "rekord_id", "meresi_pont", "gepcsoport", "idopont", "allapot",
               "ismert_gepallapot", "mertekegyseg", "detektalas", "meresi_irany", "fordulati_frekvencia_hz",
               "fordulatszam_perc", "pontszam", "maximum_amplitudo", "ertekelheto_szabalyok")}
        row.update(lehetseges_hibairanyok=" / ".join(result["lehetseges_hibairanyok"]),
                   adathibak=" / ".join(result["adathibak"]),
                   skala_indok=result["hianyzo_skala_indoka"], fordulat_indok=result["hianyzo_fordulat_indoka"],
                   pontkoz_hz=(result["frekvenciatengely"] or {}).get("pontkoz_hz"),
                   frekvenciatengely_eredete=result.get("frekvenciatengely_eredete"),
                   felteteles_frekvenciaskala=result.get("felteteles_frekvenciaskala"),
                   jellemzok_eredete=result.get("jellemzok_eredete"),
                   fordulat_eredete=result.get('fordulat_eredete'),
                   felteteles_fordulat=result.get('felteteles_fordulat'),
                   export_speed_nyers=result.get('export_speed_nyers'),
                   **result.get('fordulat_becsles', {}),
                   adatot_igenylo_vizsgalatok=" / ".join(result.get("adatot_igenylo_vizsgalatok", [])),
                   forrasok=result["forrasok"],
                   utolso_pont_hz=(result["frekvenciatengely"] or {}).get("utolso_pont_hz"),
                   adatfelhasznalasi_megjegyzesek=result.get("adatfelhasznalasi_megjegyzesek", []))
        row["meresi_irany_forrasa"] = result.get("beallitasok_eredete", {}).get("meresi_irany")
        row["hierarchia_leirasok"] = result.get("hierarchia_leirasok", [])
        summaries.append(row)
        trend = {key: row[key] for key in ("meres_id", "meresi_pont", "idopont", "mertekegyseg", "detektalas",
                                          "fordulati_frekvencia_hz", "pontkoz_hz", "maximum_amplitudo")}
        trend["osszehasonlitas"] = "Leíró adatsor; egyező mérésbeállítás és üzemállapot ellenőrzendő"
        for rule in result["szabalyok"]:
            rule_rows.append({"meres_id": record.id, "meresi_pont": record.point, "idopont": result["idopont"], **rule})
            for peak in rule["csucsok"]:
                peak_rows.append({"meres_id": record.id, "szabaly": rule["szabaly"], **peak})
                if rule["szabaly"] == "beallitas_vagy_gorbult_tengely":
                    order = round(peak["cel_hz"] / result["fordulati_frekvencia_hz"])
                    trend[f"{order}x_amplitudo"] = peak.get("amplitudo")
        trend_rows.append(trend)
        daily[(result["gepcsoport"], str(result["idopont"])[:10])].append(result)
    write_csv(destination / "eredmenyek.csv", summaries)
    write_csv(destination / "felhasznalt_jellemzok.csv", [
        {"rekord_id": result["rekord_id"], "meresi_pont": result["meresi_pont"],
         "idopont": result["idopont"], "jellemzok_eredete": result["jellemzok_eredete"],
         **result.get("felhasznalt_jellemzok", {})} for result in results])
    write_csv(destination / "szabalyonkenti_eredmenyek.csv", rule_rows)
    write_csv(destination / "csucsok.csv", peak_rows, ["meres_id", "szabaly", "cel_hz", "tures_hz", "allapot", "csucs_hz",
                                                      "amplitudo", "helyi_hatter", "hatter_arany", "index", "rend", "harmonikus", "oldal"])
    write_csv(destination / "fajlleltar.csv", inventory)
    write_csv(destination / "trend_adatok.csv", sorted(trend_rows, key=lambda r: (r["meresi_pont"], str(r["idopont"]))))
    write_csv(destination / "napi_attekintes.csv", [
        {"gepcsoport": key[0], "nap": key[1], "spektrumok": len(group),
         "tunetet_tartalmazo_spektrumok": sum(bool(r["lehetseges_hibairanyok"]) for r in group),
         "csak_leiro_vagy_hibas_spektrumok": sum(r["allapot"] in {"HIBAS_ADAT", "NEM_ERTEKELHETO", "CSAK_LEIRO_ADATOK"} for r in group),
         "jelentes": "Darabszámos áttekintés; nem teljes gépdiagnózis"} for key, group in sorted(daily.items())])
    occasions = session_rows(results)
    write_csv(destination / "alkalmi_kiertekeles.csv", occasions)
    write_csv(destination / "futas.csv", [manifest])
