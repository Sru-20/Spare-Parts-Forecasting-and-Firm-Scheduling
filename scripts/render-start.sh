#!/usr/bin/env bash
set -euo pipefail

python backend/manage.py migrate --noinput
python backend/manage.py import_catalogue
python backend/manage.py register_phase6_snapshot
python backend/manage.py bootstrap_render_admin

exec gunicorn backend.spares_api.wsgi:application \
  --bind "0.0.0.0:${PORT:-10000}" \
  --workers 1 \
  --threads 4 \
  --timeout 120
