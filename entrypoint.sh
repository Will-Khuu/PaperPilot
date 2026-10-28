#!/bin/sh
set -eu

# Fly mounts a root-owned volume over /data, hiding build-time ownership.
chown paperpilot:paperpilot /data
setpriv --reuid=paperpilot --regid=paperpilot --init-groups python deploy.py seed
exec setpriv --reuid=paperpilot --regid=paperpilot --init-groups uvicorn step7_api:app --host 0.0.0.0 --port 8080
