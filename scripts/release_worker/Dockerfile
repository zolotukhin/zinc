# Linux x64 worker runtime. GPU drivers, model pins and reference builds are
# mounted read-only by the administrator; the image contains no machine data.
FROM ubuntu:24.04@sha256:78d4744ead1f5b19b04d0df558ab182a5300903900755e515b2ca4d10eb2044d

ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
ARG RUNNER_UID=10001
ARG RUNNER_GID=10001
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git openssh-client python3 build-essential binutils procps \
    libicu74 liblttng-ust1 libssl3 libkrb5-3 zlib1g libunwind8 \
    libvulkan-dev vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/actions-runner /opt/zinc-worker /runner /run/secrets \
    && curl -fL --retry 3 "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -o /tmp/runner.tar.gz \
    && printf '%s  /tmp/runner.tar.gz\n' "$RUNNER_SHA256" | sha256sum -c - \
    && tar -xzf /tmp/runner.tar.gz -C /opt/actions-runner \
    && rm /tmp/runner.tar.gz
COPY tools/release_worker_guard.py /opt/zinc-worker/guard.py
COPY scripts/release_worker/job-started.sh scripts/release_worker/job-completed.sh /opt/zinc-worker/
COPY scripts/release_worker/entrypoint.sh /opt/zinc-worker/entrypoint.sh
RUN chmod 0555 /opt/zinc-worker/*.sh /opt/zinc-worker/guard.py
RUN git config --system --add safe.directory /reference/source
RUN groupadd --gid "$RUNNER_GID" zinc-runner \
    && useradd --no-log-init --uid "$RUNNER_UID" --gid "$RUNNER_GID" --home-dir /runner --shell /usr/sbin/nologin zinc-runner
ENV HOME=/runner \
    ZINC_RELEASE_WORKER_CONFIG=/etc/zinc-worker/worker.json \
    ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/zinc-worker/job-started.sh \
    ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/opt/zinc-worker/job-completed.sh
WORKDIR /runner
USER zinc-runner
ENTRYPOINT ["/opt/zinc-worker/entrypoint.sh"]
