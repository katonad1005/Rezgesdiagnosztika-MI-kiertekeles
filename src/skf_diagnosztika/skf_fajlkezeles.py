# Beolvassa a spektrumokat, leltárt készít az exportfájlokról, és CSV-táblákat ír.

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REQUIRED = ("Point Path", "DTS", "Unit", "Detection", "Channel", "Lines",
            "End Frequency", "Speed (Hz)", "Data")


def number(value):
    result = float(str(value).strip().replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("Nem véges szám")
    return result


def decode(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("Nem támogatott szövegkódolás")


def table(text):
    first = next((line for line in text.splitlines() if line.strip()), "")
    separator = max((";", "\t", ","), key=first.count)
    return list(csv.reader(io.StringIO(text.lstrip("\r\n")), delimiter=separator))


def timestamp(value):
    for fmt in ("%Y. %m. %d. %H:%M:%S", "%Y. %m. %d. %H:%M",
                "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value.strip(), fmt).isoformat(sep=" ")
        except ValueError:
            pass
    raise ValueError("Nem értelmezhető mérési időpont")


@dataclass
class Spectrum:
    id: str
    meta: dict
    values: list[float]
    speed: float | None
    end: float | None
    date: str
    sources: list[dict]
    errors: list[str] = field(default_factory=list)

    @property
    def point(self):
        return self.meta.get("Point Path", "")


def read_spectra(path):
    text, encoding = decode(path)
    rows = table(text)
    if not rows:
        return None, encoding, "Üres CSV"
    header = [str(x).strip() for x in rows[0]]
    if not all(key in header for key in REQUIRED):
        return None, encoding, "Nem támogatott spektrumfejléc; nem lett spektrumként értékelve"
    # Az első Unit az amplitúdóhoz, a második a folyamatértékhez tartozhat.
    ix = {key: header.index(key) for key in REQUIRED}
    if any(ix[key] >= ix["Data"] for key in REQUIRED if key != "Data"):
        return None, encoding, "A Data oszlop előtt hiányos a metaadatszerkezet"
    spectra = []
    for line, row in enumerate(rows[1:], 2):
        if not any(str(x).strip() for x in row):
            continue
        get = lambda key: row[ix[key]].strip() if ix[key] < len(row) else ""
        meta = {key: get(key) for key in REQUIRED if key != "Data"}
        errors, values, speed, end, date = [], [], None, None, ""
        for key in ("Point Path", "Unit", "Detection", "Channel"):
            if not meta[key]:
                errors.append("Hiányzó mező: " + key)
        try:
            date = timestamp(meta["DTS"])
        except ValueError as exc:
            errors.append(str(exc))
        try:
            end = number(meta["End Frequency"])
            if end <= 0:
                raise ValueError()
        except ValueError:
            errors.append("Érvénytelen End Frequency")
        if meta["Speed (Hz)"].casefold() not in {"", "n/a", "na"}:
            try:
                speed = number(meta["Speed (Hz)"])
            except ValueError:
                speed = None  # Az exportált fordulat nyers adatként marad meg.
        try:
            n = number(meta["Lines"])
            if n != int(n) or not 2 <= n <= 2_000_000:
                raise ValueError("Érvénytelen Lines")
            n = int(n)
            cells = row[ix["Data"]:]
            if len(cells) < n or any(x.strip() for x in cells[n:]):
                raise ValueError("A Lines és az amplitúdók száma eltér")
            values = [number(x) for x in cells[:n]]
            if any(x < 0 for x in values):
                raise ValueError("Negatív amplitúdó; amplitúdóspektrum szükséges")
            if not any(values):
                raise ValueError("Teljesen nulla spektrum; ellenőrizendő mérés")
        except (ValueError, OverflowError) as exc:
            errors.append("Spektrumadat-hiba: " + str(exc))
        mode = meta["Detection"].casefold().replace(" ", "").replace("-", "")
        if mode not in {"rms", "peak", "pk", "peaktopeak", "pkpk", "pktopk", "0pk"}:
            errors.append("Nem támogatott amplitúdó-detektálás: " + meta["Detection"])
        identity = [meta, values] if not errors else [meta, row[ix["Data"]:], errors]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False).encode()).hexdigest()[:24]
        spectra.append(Spectrum(digest, meta, values, speed, end, date,
                                [{"fajl": str(path), "sor": line}], errors))
    return spectra, encoding, ""


def scan(input_path, excluded=()):
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise ValueError("A bemenet nem létezik: " + str(input_path))
    paths = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    records, inventory, known = [], [], {}
    excluded = [Path(p).resolve() for p in excluded]
    for path in paths:
        if not path.is_file() or any(p == path or p in path.parents for p in excluded):
            continue
        if path.suffix.casefold() not in {".csv", ".txt", ".log"}:
            continue
        item = {"fajl": str(path), "meret_bajt": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tipus": "KIEGESZITO_SZOVEG", "meresek": 0, "hibas_sorok": 0,
                "ismetlodesei": 0, "megjegyzes": "TXT/LOG: kiegészítő információ, nem spektrum"}
        try:
            if path.suffix.casefold() != ".csv":
                _, item["kodolas"] = decode(path)
            else:
                entries, item["kodolas"], note = read_spectra(path)
                item["megjegyzes"] = note
                item["tipus"] = "EGYEB_CSV" if entries is None else "AMPLITUDOSPEKTRUM"
                for record in entries or []:
                    item["meresek"] += 1
                    item["hibas_sorok"] += bool(record.errors)
                    if record.id in known:
                        known[record.id].sources.extend(record.sources)
                        item["ismetlodesei"] += 1
                    else:
                        known[record.id] = record
                        records.append(record)
        except (ValueError, UnicodeError, csv.Error, OSError) as exc:
            item.update(tipus="OLVASASI_HIBA", megjegyzes=str(exc))
        inventory.append(item)
    return records, inventory


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False, allow_nan=False)
                                  if isinstance(value, (dict, list)) else
                                  str(value).replace(".", ",") if isinstance(value, float) else value)
                             for key, value in row.items()})
