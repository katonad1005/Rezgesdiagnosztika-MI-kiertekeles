# Az SKF-útvonalhoz hozzárendeli a fizikai gépet és annak típusát.

from pathlib import Path
from .skf_fajlkezeles import decode, table

DEFAULT_MAP = Path(__file__).resolve().parents[2] / "alkalmi_gepek.csv"


def machine_map(path):
    rows = table(decode(path)[0])
    if not rows or not {"pont_szulo","gep","geptipus"} <= set(rows[0]):
        raise ValueError("A géptérkép mezői: pont_szulo, gep, geptipus.")
    mapping, types = {}, {}
    for row in rows[1:]:
        if not any(str(v).strip() for v in row):
            continue
        data = dict(zip(rows[0],row))
        parent, machine, kind = (data.get(k,"").strip() for k in ("pont_szulo","gep","geptipus"))
        if not all((parent,machine,kind)) or parent in mapping or types.setdefault(machine,kind) != kind:
            raise ValueError("Hiányos vagy ellentmondó sor a géptérképben: " + parent)
        mapping[parent] = (machine,kind)
    return mapping
