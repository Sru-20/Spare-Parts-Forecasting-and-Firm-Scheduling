from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from django.conf import settings
from rest_framework.exceptions import NotFound, ValidationError

from backend.planning.models import ResultReference, Run


class ResultIntegrityError(ValidationError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_result_path(reference: ResultReference) -> Path:
    relative = Path(reference.relative_path)
    results_root = (settings.BASE_DIR / "results").resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ResultIntegrityError("Result reference is not a safe relative path")
    path = (settings.BASE_DIR / relative).resolve()
    if not path.is_relative_to(results_root):
        raise ResultIntegrityError("Result reference must stay inside results/")
    if not path.is_file():
        raise NotFound("The versioned result file is unavailable")
    actual_hash = file_sha256(path)
    if actual_hash != reference.file_hash:
        raise ResultIntegrityError("Result file hash does not match registered metadata")
    return path


def latest_result(run_id: Any, kind: str) -> ResultReference:
    run = Run.objects.filter(run_id=run_id, status=Run.Status.COMPLETED).first()
    if run is None:
        raise NotFound("Completed run not found")
    reference = run.results.filter(kind=kind).order_by("-version").first()
    if reference is None:
        raise NotFound(f"No {kind} result is registered for this run")
    return reference


def paged_csv_rows(
    reference: ResultReference, *, page: int, page_size: int, filters: dict[str, str]
) -> dict[str, Any]:
    path = resolve_result_path(reference)
    if page < 1 or not 1 <= page_size <= 100:
        raise ValidationError("Invalid page or page_size")
    matching = []
    total = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) > 256:
            raise ResultIntegrityError("Result file has an invalid schema")
        allowed_filters = set(reader.fieldnames) & {
            "part_id", "region", "week", "status", "criticality", "category"
        }
        unknown = set(filters) - allowed_filters
        if unknown:
            raise ValidationError(
                {"filters": f"Unsupported result filters: {', '.join(sorted(unknown))}"}
            )
        start = (page - 1) * page_size
        stop = start + page_size
        for row in reader:
            if all(row.get(key) == value for key, value in filters.items()):
                if total >= start and total < stop:
                    matching.append(row)
                total += 1
                if total > 2_000_000:
                    raise ResultIntegrityError("Result exceeds the supported row limit")
    return {
        "run_id": str(reference.run_id),
        "result_kind": reference.kind,
        "result_version": reference.version,
        "page": page,
        "page_size": page_size,
        "total": total,
        "results": matching,
    }


def read_csv_records(reference: ResultReference) -> tuple[list[str], list[dict[str, str]]]:
    path = resolve_result_path(reference)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) > 256:
            raise ResultIntegrityError("Result file has an invalid schema")
        rows = list(reader)
    if len(rows) > 2_000_000:
        raise ResultIntegrityError("Result exceeds the supported row limit")
    return list(reader.fieldnames), rows


def json_result(reference: ResultReference) -> dict[str, Any]:
    path = resolve_result_path(reference)
    if path.suffix.casefold() != ".json":
        raise ResultIntegrityError("JSON result endpoint requires a JSON result")
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ResultIntegrityError("JSON result exceeds the supported size")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultIntegrityError("Registered JSON result is malformed") from error
    if not isinstance(content, (dict, list)):
        raise ResultIntegrityError("Registered JSON result has an unsupported shape")
    return {
        "run_id": str(reference.run_id),
        "result_kind": reference.kind,
        "result_version": reference.version,
        "results": content,
    }
