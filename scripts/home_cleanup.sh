#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--days N] [--exec]

By default this performs a dry-run and prints what would be deleted.
Use --exec to actually remove files.
Targets: common cache and tmp directories in HOME (only if they exist).
EOF
}

DAYS=30
EXEC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)
      DAYS="$2"; shift 2;;
    --exec)
      EXEC=1; shift;;
    -h|--help)
      usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

TARGETS=(
  "$HOME/conda_pkgs"
  "$HOME/pip_cache"
  "$HOME/openpi_cache"
  "$HOME/tmp"
  "$HOME/.cache"
  "$HOME/sae_checkpoints"
  "$HOME/sae_interpretations"
)

echo "Home cleanup: dry-run by default. Days threshold: $DAYS"
if [[ $EXEC -eq 1 ]]; then
  echo "*** EXECUTION MODE: will delete files older than $DAYS days"
else
  echo "*** DRY RUN: no files will be deleted. Use --exec to delete."
fi

TOTAL_WOULD_DELETE=0

for d in "${TARGETS[@]}"; do
  if [[ -e "$d" ]]; then
    echo "\nScanning: $d"
    du -sh "$d" 2>/dev/null || true
    echo "Files older than $DAYS days (first 20 shown):"
    find "$d" -mindepth 1 -mtime +$DAYS -print | head -n 20 || true
    if [[ $EXEC -eq 1 ]]; then
      # compute size to be removed
      size_bytes=$(find "$d" -mindepth 1 -mtime +$DAYS -printf "%s\n" 2>/dev/null | awk '{s+=$1} END{print s+0}') || size_bytes=0
      if [[ -n "$size_bytes" && "$size_bytes" -gt 0 ]]; then
        echo "Removing contents older than $DAYS days from $d"
        find "$d" -mindepth 1 -mtime +$DAYS -print0 | xargs -0 rm -rf || true
        echo "Removed approx $(awk -v b=$size_bytes 'BEGIN{printf "%.2f MB", b/1024/1024}')"
        TOTAL_WOULD_DELETE=$((TOTAL_WOULD_DELETE + size_bytes))
      else
        echo "Nothing to remove here."
      fi
    else
      would_size=$(find "$d" -mindepth 1 -mtime +$DAYS -printf "%s\n" 2>/dev/null | awk '{s+=$1} END{print s+0}') || would_size=0
      echo "Would delete approx: $(awk -v b=$would_size 'BEGIN{printf "%.2f MB", b/1024/1024}')"
      TOTAL_WOULD_DELETE=$((TOTAL_WOULD_DELETE + would_size))
    fi
  else
    echo "\nTarget not found: $d"
  fi
done

echo "\nSummary:"
echo "Total bytes targeted: $TOTAL_WOULD_DELETE"
echo "Approx: $(awk -v b=$TOTAL_WOULD_DELETE 'BEGIN{printf "%.2f MB", b/1024/1024}')"

if [[ $EXEC -eq 0 ]]; then
  echo "\nTo actually delete these files run: $0 --days $DAYS --exec"
fi
