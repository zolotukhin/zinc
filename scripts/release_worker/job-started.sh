#!/usr/bin/env bash
set -eu
exec /usr/bin/python3 -I /opt/zinc-worker/guard.py --policy /etc/zinc-worker/policy.json
