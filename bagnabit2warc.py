#!/usr/bin/env python3
"""
bagnabit2warc.py — Convert a bag-nabit dataset stored in a ZIP into a full-content WARC.

Behavior:
- Reads a bag-nabit-style ZIP containing data/headers.warc and, optionally, data/files/*.
- Emits a single JSON log line to stderr per input file.
- Creates a WARC with:
  - a warcinfo record
  - an optional metadata record from data/signed-metadata.json
  - original request/response/other records from data/headers.warc
  - reconstructed response records for revisit records that point to payloads in data/files/*

Notes:
- This is not a generic BagIt -> WARC converter. It is specific to the bag-nabit layout.
- If the output path is omitted, it defaults to input.zip -> input.warc.gz
- For large payloads, the code passes length=zipinfo.file_size to warcio to avoid
  spooling large temporary files to /tmp during digest calculation.

Requires:
    pip install warcio
"""

from __future__ import annotations

import argparse
import email.utils
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Tuple

from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter


PROFILE_RE = re.compile(r'^\s*file-content\s*;\s*filename\s*=\s*"([^"]+)"\s*$')


@dataclass
class BagnabitZipPaths:
    zip_path: Path
    bag_prefix: str
    headers_warc: str
    files_dir: str
    signed_metadata: Optional[str]
    has_files_dir: bool


@dataclass
class Stats:
    warcinfo_records_written: int = 0
    metadata_records_written: int = 0

    headers_records_total: int = 0
    headers_requests: int = 0
    headers_responses: int = 0
    headers_revisits: int = 0
    headers_other: int = 0

    requests_written: int = 0
    original_responses_written: int = 0
    responses_reconstructed: int = 0
    other_warc_written: int = 0
    revisits_passed_through: int = 0

    payload_missing: int = 0
    revisits_without_profile: int = 0

    warnings_count: int = 0
    errors_count: int = 0


@dataclass
class RunLog:
    input_file: str
    output_file: str
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    stats: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _emit_json_log(runlog: RunLog) -> None:
    sys.stderr.write(json.dumps(asdict(runlog), ensure_ascii=False, sort_keys=True) + "\n")
    sys.stderr.flush()


def _add_warning(runlog: RunLog, stats: Stats, msg: str) -> None:
    stats.warnings_count += 1
    runlog.warnings.append(msg)


def _add_error(runlog: RunLog, stats: Stats, msg: str) -> None:
    stats.errors_count += 1
    runlog.errors.append(msg)


def _parse_warc_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if value.endswith("Z") and "T" in value:
        return value
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _norm_zip_name(name: str) -> str:
    return name.replace("\\", "/")


def _default_output_path(bag_zip: Path) -> Path:
    if bag_zip.suffix.lower() == ".zip":
        return bag_zip.with_suffix(".warc.gz")
    return Path(str(bag_zip) + ".warc.gz")


def _discover_zip_paths(zf: zipfile.ZipFile, zip_path: Path) -> BagnabitZipPaths:
    names = [_norm_zip_name(n) for n in zf.namelist()]
    headers_matches = [n for n in names if n.endswith("data/headers.warc")]

    if not headers_matches:
        raise FileNotFoundError("ZIP does not contain data/headers.warc")

    headers_matches.sort(key=len)
    headers_warc = headers_matches[0]

    bag_prefix = headers_warc[: -len("data/headers.warc")]
    if bag_prefix and not bag_prefix.endswith("/"):
        bag_prefix += "/"

    files_dir = f"{bag_prefix}data/files/"
    signed_metadata = f"{bag_prefix}data/signed-metadata.json"
    if signed_metadata not in names:
        signed_metadata = None

    has_files_dir = any(n.startswith(files_dir) for n in names)

    return BagnabitZipPaths(
        zip_path=zip_path,
        bag_prefix=bag_prefix,
        headers_warc=headers_warc,
        files_dir=files_dir,
        signed_metadata=signed_metadata,
        has_files_dir=has_files_dir,
    )


def _profile_filename(rec) -> Optional[str]:
    prof = rec.rec_headers.get_header("WARC-Profile")
    if not prof:
        return None
    m = PROFILE_RE.match(prof)
    if not m:
        return None
    return m.group(1)


def _safe_zip_join(prefix: str, relative: str) -> str:
    rel = PurePosixPath(_norm_zip_name(relative))
    if rel.is_absolute():
        raise ValueError(f"Absolute path in WARC-Profile: {relative}")

    combined = PurePosixPath(prefix) / rel
    parts: List[str] = []
    for p in combined.parts:
        if p == "..":
            raise ValueError(f"Path traversal outside bag: {relative}")
        if p == ".":
            continue
        parts.append(p)

    return "/".join(parts)


def _iter_headers_records(zf: zipfile.ZipFile, headers_name: str) -> Iterator:
    with zf.open(headers_name, "r") as fh:
        for rec in ArchiveIterator(fh):
            yield rec


def _parse_metadata_kv(raw: str) -> Tuple[str, str]:
    if ":" not in raw:
        raise ValueError(f'Invalid --metadata {raw!r}; expected "key: value"')
    key, value = raw.split(":", 1)
    key = key.strip()
    value = value.lstrip()
    if not key:
        raise ValueError(f'Invalid --metadata {raw!r}; key is empty')
    return key, value


def _build_warcinfo_fields(bag: BagnabitZipPaths, overrides: List[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {
        "software": "bagnabit2warc (warcio)",
        "format": "WARC File Format 1.0",
        "conformsTo": "https://iipc.github.io/warc-specifications/specifications/warc-format/warc-1.0/",
        "created": _iso_z(_utcnow()),
        "source": str(bag.zip_path),
    }

    for raw in overrides:
        k, v = _parse_metadata_kv(raw)
        fields[k] = v

    return fields


def _write_warcinfo(
    writer: WARCWriter,
    bag: BagnabitZipPaths,
    overrides: List[str],
    stats: Stats,
) -> None:
    fields = _build_warcinfo_fields(bag, overrides)
    payload = BytesIO(
        ("\r\n".join(f"{k}: {v}" for k, v in fields.items()) + "\r\n").encode("utf-8")
    )

    rec = writer.create_warc_record(
        "",
        "warcinfo",
        payload=payload,
        warc_headers_dict={"WARC-Date": _iso_z(_utcnow())},
    )
    writer.write_record(rec)
    stats.warcinfo_records_written += 1


def _write_signed_metadata(
    writer: WARCWriter,
    zf: zipfile.ZipFile,
    bag: BagnabitZipPaths,
    stats: Stats,
    runlog: RunLog,
) -> None:
    if not bag.signed_metadata:
        return

    try:
        with zf.open(bag.signed_metadata, "r") as fh:
            raw = fh.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        _add_warning(runlog, stats, f"Failed to read/parse {bag.signed_metadata} from ZIP: {e}")
        return

    target_uri = ""
    if isinstance(data, dict):
        maybe = data.get("url")
        if isinstance(maybe, str) and maybe.startswith(("http://", "https://")):
            target_uri = maybe
    if not target_uri:
        target_uri = f"urn:uuid:{os.urandom(16).hex()}"

    rec = writer.create_warc_record(
        target_uri,
        "metadata",
        payload=BytesIO(raw),
        warc_headers_dict={
            "WARC-Date": _iso_z(_utcnow()),
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    writer.write_record(rec)
    stats.metadata_records_written += 1


def convert_zip_bag_to_warc(
    bag_zip: Path,
    out_warc: Path,
    include_signed_metadata: bool = True,
    metadata_overrides: Optional[List[str]] = None,
    strict: bool = False,
) -> RunLog:
    metadata_overrides = metadata_overrides or []
    stats = Stats()

    start_dt = _utcnow()
    t0 = time.monotonic()

    runlog = RunLog(
        input_file=str(bag_zip),
        output_file=str(out_warc),
        input_size_bytes=bag_zip.stat().st_size if bag_zip.exists() else None,
        start_time_utc=_iso_z(start_dt),
    )

    try:
        if not bag_zip.is_file():
            raise FileNotFoundError(f"No such zip file: {bag_zip}")

        out_warc.parent.mkdir(parents=True, exist_ok=True)
        gzip_output = out_warc.suffix == ".gz"

        with zipfile.ZipFile(bag_zip, "r") as zf:
            bag = _discover_zip_paths(zf, bag_zip)
            names_set = set(_norm_zip_name(n) for n in zf.namelist())

            if not bag.has_files_dir:
                _add_warning(runlog, stats, f"ZIP has no payload directory entries under {bag.files_dir}")

            with out_warc.open("wb") as out_fh:
                writer = WARCWriter(out_fh, gzip=gzip_output)

                try:
                    _write_warcinfo(writer, bag, metadata_overrides, stats)
                except Exception as e:
                    if strict:
                        raise
                    _add_error(runlog, stats, f"Failed to write warcinfo record: {e}")

                if include_signed_metadata:
                    _write_signed_metadata(writer, zf, bag, stats, runlog)

                try:
                    for rec in _iter_headers_records(zf, bag.headers_warc):
                        stats.headers_records_total += 1

                        warc_type = rec.rec_headers.get_header("WARC-Type")
                        uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
                        warc_date = _parse_warc_date(rec.rec_headers.get_header("WARC-Date")) or _iso_z(_utcnow())

                        if warc_type == "request":
                            stats.headers_requests += 1
                            writer.write_record(rec)
                            stats.requests_written += 1
                            continue

                        if warc_type == "response":
                            stats.headers_responses += 1
                            writer.write_record(rec)
                            stats.original_responses_written += 1
                            continue

                        if warc_type == "revisit":
                            stats.headers_revisits += 1

                            filename = _profile_filename(rec)
                            if not filename:
                                stats.revisits_without_profile += 1
                                writer.write_record(rec)
                                stats.revisits_passed_through += 1
                                continue

                            try:
                                payload_entry = _safe_zip_join(f"{bag.bag_prefix}data", filename)
                            except Exception as e:
                                msg = f"Bad payload filename in WARC-Profile ({filename}): {e}"
                                if strict:
                                    raise ValueError(msg)
                                _add_warning(runlog, stats, msg)
                                stats.payload_missing += 1
                                writer.write_record(rec)
                                stats.revisits_passed_through += 1
                                continue

                            if payload_entry not in names_set:
                                msg = f"Missing payload ZIP entry referenced by headers.warc: {payload_entry}"
                                if strict:
                                    raise FileNotFoundError(msg)
                                _add_warning(runlog, stats, msg)
                                stats.payload_missing += 1
                                writer.write_record(rec)
                                stats.revisits_passed_through += 1
                                continue

                            warc_headers: Dict[str, str] = {"WARC-Date": warc_date}
                            if uri:
                                warc_headers["WARC-Target-URI"] = uri

                            for h in ("WARC-Record-ID", "WARC-IP-Address", "WARC-Warcinfo-ID"):
                                v = rec.rec_headers.get_header(h)
                                if v:
                                    warc_headers[h] = v

                            zipinfo = zf.getinfo(payload_entry)

                            with zf.open(zipinfo, "r") as payload_fh:
                                response_rec = writer.create_warc_record(
                                    uri or f"urn:uuid:{os.urandom(16).hex()}",
                                    "response",
                                    payload=payload_fh,
                                    length=zipinfo.file_size,
                                    http_headers=rec.http_headers,
                                    warc_headers_dict=warc_headers,
                                )
                                writer.write_record(response_rec)
                                stats.responses_reconstructed += 1

                            continue

                        stats.headers_other += 1
                        writer.write_record(rec)
                        stats.other_warc_written += 1

                except Exception as e:
                    if strict:
                        raise
                    _add_error(runlog, stats, f"Unexpected error during conversion: {e}")

        runlog.success = (stats.errors_count == 0)

    except Exception as e:
        _add_error(runlog, stats, f"Fatal conversion error: {e}")
        runlog.success = False

    end_dt = _utcnow()
    runlog.end_time_utc = _iso_z(end_dt)
    runlog.duration_seconds = round(time.monotonic() - t0, 6)
    runlog.stats = asdict(stats)

    if stats.headers_records_total == 0:
        _add_warning(
            runlog,
            stats,
            "headers.warc contained 0 records; output will only include warcinfo (and optional metadata).",
        )
        runlog.stats = asdict(stats)

    if out_warc.exists():
        try:
            runlog.output_size_bytes = out_warc.stat().st_size
        except Exception:
            runlog.output_size_bytes = None

    return runlog


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a bag-nabit ZIP dataset into a full-content WARC."
    )
    parser.add_argument("bag_zip", type=Path, help="Path to bag-nabit ZIP input")
    parser.add_argument("out_warc", type=Path, nargs="?", help="Optional output WARC path")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help='Repeatable. Add/overwrite warcinfo fields. Format: --metadata "key: value"',
    )
    parser.add_argument(
        "--no-signed-metadata",
        action="store_true",
        help="Do not add data/signed-metadata.json as a WARC metadata record",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on recoverable issues such as missing payload files",
    )
    args = parser.parse_args(argv)

    bag_zip = args.bag_zip
    out_warc = args.out_warc if args.out_warc else _default_output_path(bag_zip)

    try:
        for raw in args.metadata:
            _parse_metadata_kv(raw)
    except Exception as e:
        runlog = RunLog(
            input_file=str(bag_zip),
            output_file=str(out_warc),
            input_size_bytes=bag_zip.stat().st_size if bag_zip.exists() else None,
            start_time_utc=_iso_z(_utcnow()),
            end_time_utc=_iso_z(_utcnow()),
            duration_seconds=0.0,
            success=False,
            stats=asdict(Stats(errors_count=1)),
            warnings=[],
            errors=[str(e)],
        )
        _emit_json_log(runlog)
        return 1

    runlog = convert_zip_bag_to_warc(
        bag_zip=bag_zip,
        out_warc=out_warc,
        include_signed_metadata=not args.no_signed_metadata,
        metadata_overrides=args.metadata,
        strict=args.strict,
    )
    _emit_json_log(runlog)
    return 0 if runlog.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
