# A kiválasztott légkezelőmérést a csomaghoz mellékelt modellel értékeli ki.

"""Légkezelő-spektrumok kiértékelése a csomagolt modellel."""
import argparse
from datetime import datetime
from pathlib import Path
import tempfile

from alkalmi_mi import ROOT, collect, inventory, load_config, load_point_settings, main as model_main, read_csv, write_csv


def choose_files():
    import tkinter as tk
    from tkinter.filedialog import askopenfilenames
    window = tk.Tk()
    window.withdraw()
    try:
        return [Path(name) for name in askopenfilenames(
            title="Légkezelők SKF amplitúdóspektrumai – vállalati Hz-skála",
            filetypes=[("SKF spektrum CSV", "*.csv")])]
    finally:
        window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, help="SKF légkezelő-spektrum CSV")
    args = parser.parse_args()
    try:
        paths = args.csv or choose_files()
        if not paths:
            return 0
        if any("phase" in p.stem.casefold() for p in paths):
            raise ValueError("Amplitúdóspektrumot válassz a fázisexport helyett.")
        results = collect(paths, ROOT/"alkalmi_gepek.csv", load_config(ROOT/"skf_beallitasok.json"),
                          load_point_settings(ROOT/"skf_pontbeallitasok.csv"), legkezelo_export=True)
        rows = inventory(results)
        folder = ROOT/"outputs"/"legkezelo_proba"
        folder.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=folder))
        write_csv(run/"alkalmak.csv", rows)
        status = model_main(["becsles", "--mappa", str(run), "--modell",
                             str(ROOT/"legkezelo_modell"/"alkalmi_modell.joblib")])
        if status == 0:
            for row in read_csv(run/"alkalmi_becslesek.csv").itertuples():
                print(f"{row.gep} | {row.kezdet} | {row.becsles}")
        return status
    except (ValueError, KeyError, ImportError, OSError) as exc:
        print(f"Kiértékelési hiba: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
