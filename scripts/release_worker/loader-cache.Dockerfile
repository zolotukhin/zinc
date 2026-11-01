# Generate this cache using this exact base image and its read-only driver and
# runtime mounts. Keep the generated cache in a private build context.
ARG ZINC_WORKER_BASE=zinc-release-worker:base
FROM ${ZINC_WORKER_BASE}
COPY --chown=0:0 --chmod=0644 ld.so.cache /etc/ld.so.cache
