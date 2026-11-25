# Az előkészített spektrumokra alkalmazza a diagnosztikai szabályokat, majd elindítja az eredmények mentését.

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import uuid

from .skf_beallitasok import load_config, load_point_settings, resolve
from .skf_elokeszites import automatic_context, from_main
from .skf_hierarchia import lookup
from .skf_geptipus import DEFAULT_MAP, machine_map
from .skf_jelentes import write_reports
from .skf_szabalyok import VERSION, evaluate
from fordulatszam import MODSZER


def assess_prepared(prepared, config, point_settings, *, supplied_axes, known_faulty=False, machines=None):
    records, results = [], []
    hierarchy_cache = {}
    if machines is None:
        machines = machine_map(DEFAULT_MAP) if DEFAULT_MAP.is_file() else {}
    for record, axis, features, axis_origin, metadata in prepared:
        settings, origins, profiles, errors = resolve(record.point, config, point_settings, record.date)
        direction, descriptions = lookup(record, hierarchy_cache)
        if direction and origins["meresi_irany"] == "ismeretlen":
            settings["meresi_irany"] = direction
            origins["meresi_irany"] = "Hierarchy pontleírás"
        if metadata and not settings.get("gep"):
            machine = metadata.get("gepegyseg_utvonal")
            if isinstance(machine, str) and machine.strip():
                settings["gep"] = machine
                origins["gep"] = "fo_program_gepazonositasa"
        if settings.get("villanymotor") is None:
            if re.search(r"(?:^|[\\/_ -])MOTOR(?:$|[\\/_ -])", record.point, re.I):
                settings["villanymotor"] = True
                origins["villanymotor"] = "pontelnevezesbol_felismert"
            elif re.search(r"(?:^|[\\/_ -])(?:TENGELY|TD|TN|VENTI|VENTILATOR)(?:$|[\\/_ -])", record.point, re.I):
                settings["villanymotor"] = False
                origins["villanymotor"] = "pontelnevezesbol_felismert"
        context = automatic_context(record, settings, config["csucskereses"], axis=axis,
                                    axis_supplied=supplied_axes, axis_origin=axis_origin,
                                    features=features,
                                    feature_origin="fo_program_jellemzokinyerese" if supplied_axes else "onallo_jellemzokinyeres")
        result = evaluate(record, settings, config["csucskereses"], errors, known_faulty, context=context)
        result["hierarchia_leirasok"] = descriptions
        parent = record.point.rsplit(chr(92),1)[0]
        if parent in machines:
            result["fizikai_gep"], result["geptipus"] = machines[parent]
        result["csatorna"] = record.meta.get("Channel")
        result["beallitasok_eredete"] = origins
        result["alkalmazott_profilok"] = profiles
        records.append(record)
        results.append(result)
    return records, results


def finalize(records, results, inventory, output_parent, manifest):
    now = datetime.now().astimezone()
    destination = Path(output_parent) / (now.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    manifest.update(idopont=now.isoformat(), verzio=VERSION,
                    fordulat_modszer=MODSZER,
                    adatfolyam="beolvasas -> frekvenciatengely -> vallalati_fordulatbecsles -> jellemzok -> SKF-reszvizsgalatok",
                    technikai_kuszobok_eredete="Saját, állítható csúcskeresési paraméterek; nem SKF riasztási határok",
                    gepallapot_dontes="A tünet és a feltételes eredmény nem igazolt állapotcímke")
    write_reports(destination, records, results, inventory, manifest)
    print("\nSKF-kiértékelés:")
    for state, count in sorted(Counter(r["allapot"] for r in results).items()):
        print(f"  {state}: {count}")
    print("Jelentés: " + str((destination / "alkalmi_kiertekeles.csv").resolve()))
    conditional = sum(r["felteteles_frekvenciaskala"] for r in results)
    if conditional:
        print(f"Feltételes skálaprofillal feldolgozott spektrum: {conditional}; az eredet a jelentésben látható.")
    print("A tünet hiánya nem igazol megfelelő gépállapotot.")
    return destination


def run_prepared(meta, spectra, axes, features, source, output_parent, *, config_path, points_path):
    """A főprogram ezt hívja; nincs scan(), input() vagy második CSV-beolvasás."""
    config = load_config(config_path)
    points = load_point_settings(points_path)
    prepared = from_main(meta, spectra, axes, features, source)
    records, results = assess_prepared(prepared, config, points, supplied_axes=True)
    inventory = [{"fajl": str(Path(source).resolve()), "tipus": "FO_PROGRAM_ELŐFELDOLGOZOTT_ADATAI",
                  "meret_bajt": Path(source).stat().st_size if Path(source).exists() else None,
                  "meresek": len(records), "hibas_sorok": sum(bool(r.errors) for r in records),
                  "ismetlodesei": 0, "megjegyzes": "A főprogram metaadatait, tengelyeit, spektrumait és jellemzőit kapta meg; nincs újraolvasás"}]
    manifest = {"bemenet": str(Path(source).resolve()), "konfig_fajl": str(Path(config_path).resolve()),
                "konfiguracio": config, "pontbeallitasok": points,
                "jellemzok_forrasa": "fo_program_jellemzok.csv", "export_ujraolvasva": False}
    return finalize(records, results, inventory, output_parent, manifest)
