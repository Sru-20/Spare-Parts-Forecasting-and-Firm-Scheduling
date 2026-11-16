from django.core.management.base import BaseCommand

from backend.assistant.evaluation import write_evaluation
from django.conf import settings


class Command(BaseCommand):
    help = "Evaluate the deterministic Ask AI fallback on labeled dev/test requests."

    def handle(self, *args, **options):
        requests_path, report_path, metrics = write_evaluation(
            settings.BASE_DIR / "results",
            settings.BASE_DIR / "reports",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"requests={metrics['total_samples']} "
                f"test_intent_accuracy={metrics['test_intent_accuracy']:.3f} "
                f"test_entity_accuracy={metrics['test_entity_accuracy']:.3f}"
            )
        )
        self.stdout.write(f"Dataset: {requests_path}")
        self.stdout.write(f"Report: {report_path}")
