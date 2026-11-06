#!/usr/bin/env bash
# Source on an authorized CUDA worker. Roles are configured in the worker's
# environment; neither a workstation's identifiers nor a fallback GPU belong
# in a reusable experiment. This function never probes or opens a GPU.
zinc_select_cuda_device() {
  local selected="${ZINC_GPU:-${CUDA_VISIBLE_DEVICES:-}}"
  case "$selected" in
    4090) selected="${ZINC_GPU_4090:-}" ;;
    5090) selected="${ZINC_GPU_5090:-}" ;;
  esac
  if [[ -z "$selected" ]]; then
    echo 'Set ZINC_GPU to one CUDA device, or configure ZINC_GPU_4090/ZINC_GPU_5090 for the selected role.' >&2
    return 1
  fi
  if [[ ! "$selected" =~ ^[0-9]+$ && ! "$selected" =~ ^GPU-[[:xdigit:]-]+$ && ! "$selected" =~ ^MIG-[[:xdigit:]/-]+$ ]]; then
    echo 'Expected one CUDA ordinal, GPU identifier, or MIG identifier.' >&2
    return 1
  fi
  export CUDA_VISIBLE_DEVICES="$selected" ZINC_GPU="$selected"
}
