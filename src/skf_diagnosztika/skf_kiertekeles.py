# A parancssori beállításokkal beolvassa az exportot és elindítja a tünetvizsgálatot.

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from .skf_beallitasok import (load_config, load_point_settings, point_template,
                        resolve, save_config, wizard)
from .skf_fajlkezeles import scan
from .skf_feldolgozas import assess_prepared, finalize
from .skf_szabalyok import VERSION

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description="SKF-spektrumok szabályalapú kiértékelése, gépi tanulás nélkül.")
    p.add_argument("--bemenet", type=Path, default=ROOT / "data", help="Exportmappa vagy egy spektrum-CSV")
    p.add_argument("--kimenet", type=Path, default=ROOT / "outputs" / "skf_kiertekeles", help="A futásmappák szülőmappája")
    p.add_argument("--konfig", type=Path, default=ROOT / "skf_beallitasok.json")
    p.add_argument("--pontok", type=Path, default=ROOT / "skf_pontbeallitasok.csv")
    p.add_argument("--beallitas", action="store_true", help="Opcionális frekvenciaskála-beállítás; fordulatot nem kérdez")
    p.add_argument("--pont-sablon", action="store_true", help="Opcionális pontbeállítási sablon készítése")
    p.add_argument("--ismerten-hibas", action="store_true", help="Külön feljegyzi, hogy a felhasználó minden bemeneti mérést hibásnak jelölt")
    p.add_argument("--kepek-nelkul", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--pelda", action="store_true", help="A mellékelt mesterséges tesztspektrumokon fut")
    p.add_argument("--verzio", action="version", version=VERSION)
    return p


def run(args):
    if args.pelda:
        args.bemenet = ROOT / "pelda_skf" / "data"
        args.konfig = ROOT / "pelda_skf" / "beallitasok.json"
        args.pontok = ROOT / "pelda_skf" / "pontbeallitasok.csv"
    if not args.bemenet.exists():
        raise ValueError("A bemenet nem létezik: " + str(args.bemenet))
    args.konfig.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.konfig)
    if args.beallitas:
        config = wizard(copy.deepcopy(config))
        save_config(args.konfig, config)
    records, inventory = scan(args.bemenet, excluded=[args.kimenet.resolve(), args.pontok.resolve()])
    if not records:
        print("Nem találtam támogatott SKF-amplitúdóspektrumot.")
        for file in inventory:
            print(f'  {Path(file["fajl"]).name}: {file["tipus"]} — {file["megjegyzes"]}')
        return 2
    args.pontok.parent.mkdir(parents=True, exist_ok=True)
    load_point_settings(args.pontok)  # A meglévő pontbeállításokat a sablon írása előtt ellenőrzöm.
    if args.pont_sablon or args.beallitas:
        point_template(args.pontok, records)
    point_settings = load_point_settings(args.pontok)
    print(f"Feldolgozás: {len(records)} egyedi spektrum, {len(inventory)} exportfájl.")
    prepared = [(record, None, None, "", None) for record in records]
    records, results = assess_prepared(prepared, config, point_settings, supplied_axes=False,
                                      known_faulty=args.ismerten_hibas)
    manifest = {"bemenet": str(args.bemenet.resolve()), "konfig_fajl": str(args.konfig.resolve()),
                "pontbeallitas_fajl": str(args.pontok.resolve()), "konfiguracio": config,
                "pontbeallitasok": point_settings, "ismerten_hibas_kapcsolo": args.ismerten_hibas,
                "jellemzok_forrasa": "beolvasott_exportbol_automatikusan_szamitva", "export_ujraolvasva": False}
    finalize(records, results, inventory, args.kimenet, manifest)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("HIBA: " + str(exc), file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("A futás megszakadt.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
