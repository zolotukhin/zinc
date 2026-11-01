#!/usr/bin/env bash
set -eu
if [ "$(id -u)" = 0 ]; then
  printf 'The release runner must use a dedicated non-root UID.\n' >&2
  exit 1
fi
if [ ! -f .runner ]; then
  cp -R --no-preserve=ownership /opt/actions-runner/. /runner/
  export ACTIONS_RUNNER_INPUT_TOKEN
  ACTIONS_RUNNER_INPUT_TOKEN=$(cat /run/secrets/registration-token)
  test -n "$ACTIONS_RUNNER_INPUT_TOKEN"
  ./config.sh --unattended --disableupdate \
    --url "${ZINC_RUNNER_REPOSITORY_URL:?}" --name "${ZINC_RUNNER_NAME:?}" \
    --labels "${ZINC_RUNNER_LABELS:?}" --work _work
  unset ACTIONS_RUNNER_INPUT_TOKEN
fi
exec nice -n 10 ./run.sh
