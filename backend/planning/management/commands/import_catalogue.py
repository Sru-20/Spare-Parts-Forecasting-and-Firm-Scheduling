from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from backend.planning.models import Part
from src.data.loader import ROOT, load_catalogue


class Command(BaseCommand):
    help = "Import the schema-validated D3 catalogue without changing source data."

    def add_arguments(self, parser):
        parser.add_argument("--source", default=str(ROOT / "data" / "03_spare_parts_catalogue.csv"))

    def handle(self, *args, **options):
        source = Path(options["source"]).resolve()
        try:
            catalogue = load_catalogue(source)
        except (OSError, ValueError) as error:
            raise CommandError(f"Unable to load validated catalogue: {error}") from error
        file_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        batch = []
        for row in catalogue.to_dict(orient="records"):
            batch.append(
                Part(
                    part_id=str(row["part_id"]),
                    part_number=str(row.get("part_number") or ""),
                    part_name=str(row["part_name"]),
                    component=str(row.get("component") or ""),
                    category=str(row.get("part_category") or ""),
                    material=str(row.get("material") or ""),
                    vehicle_type=str(row.get("vehicle_type") or ""),
                    criticality=str(row.get("criticality") or ""),
                    lifecycle_status=str(row.get("lifecycle_status") or ""),
                    source_hash=file_hash,
                )
            )
        for part in batch:
            Part.objects.update_or_create(
                part_id=part.part_id,
                defaults={
                    "part_number": part.part_number,
                    "part_name": part.part_name,
                    "component": part.component,
                    "category": part.category,
                    "material": part.material,
                    "vehicle_type": part.vehicle_type,
                    "criticality": part.criticality,
                    "lifecycle_status": part.lifecycle_status,
                    "source_hash": file_hash,
                },
            )
        self.stdout.write(self.style.SUCCESS(f"Imported {len(batch)} catalogue parts"))
