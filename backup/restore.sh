#!/usr/bin/env bash
# backup/restore.sh — восстановление из архива, созданного backup.sh
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups}"
TARGET_DB="${TARGET_DB:-$ROOT_DIR/data/ai_trader.db}"
DB_TYPE="${DB_TYPE:-sqlite}"
ARCHIVE="${1:-}"

if [[ -z "$ARCHIVE" ]]; then
  echo "Usage: $0 <archive-file>" >&2
  echo "  archive-file — путь до .tar.gz/.tar.zst, созданного backup.sh" >&2
  exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 1
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "[restore] unpacking $ARCHIVE"
tar -xf "$ARCHIVE" -C "$workdir"

if [[ "$DB_TYPE" == "postgres" ]]; then
  dump_file="$(find "$workdir" -name '*.dump' -print -quit)"
  if [[ -z "$dump_file" ]]; then
    echo "pg_dump archive not found in $ARCHIVE" >&2
    exit 1
  fi
  echo "[restore] restoring PostgreSQL dump $dump_file"
  pg_restore --clean --if-exists --create --no-owner "$dump_file"
else
  backup_file="$(find "$workdir" -name '*.db' -print -quit)"
  if [[ -z "$backup_file" ]]; then
    backup_file="$(find "$workdir" -name '*.sqlite' -print -quit)"
  fi
  if [[ -z "$backup_file" ]]; then
    echo "SQLite backup not found in $ARCHIVE" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$TARGET_DB")"
  echo "[restore] restoring SQLite DB → $TARGET_DB"
  cp "$backup_file" "$TARGET_DB"
fi

echo "[restore] done"
