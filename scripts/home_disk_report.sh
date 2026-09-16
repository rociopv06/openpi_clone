#!/usr/bin/env bash
set -euo pipefail

echo "HOME path: $HOME"
echo "Top 30 entries in $HOME by size:"
du -sh "$HOME"/* 2>/dev/null | sort -hr | head -n 30

echo "\nTop 30 largest files under $HOME:"
find "$HOME" -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -n 30 | awk '{printf "%10.2f MB  %s\n", $1/1024/1024, substr($0, index($0,$2))}'

echo "\nDisk free space:" 
df -h "$HOME"
