#!/usr/bin/env bash
# backup/backup.sh
# Ежедневные бэкапы БД и logs/ с ротацией.
# Поддержка: PostgreSQL (pg_dump -Fc) и SQLite (sqlite3 .backup)
# Опционально: отправка в S3 (AWS CLI), сжатие, хэши.
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация (можно переопределить через .env или переменные окружения)
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/configs/.env}"

# Куда складывать бэкапы
BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups}"
# Что дополнительно архивировать (каталог логов)
LOGS_DIR="${LOGS_DIR:-$ROOT_DIR/logs}"

# Тип БД: "postgres" | "sqlite"
DB_TYPE="${DB_TYPE:-postgres}"

# Параметры PostgreSQL (если DB_TYPE=postgres)
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-trader}"
PGDATABASE="${PGDATABASE:-ai_trader}"
# Если нужен пароль: экспортируй PGPASSWORD в .env или окружение

# Путь к SQLite (если DB_TYPE=sqlite)
SQLITE_PATH="${SQLITE_PATH:-$ROOT_DIR/ai_trader.db}"

# Ротация (хранить N дней)
RETENTION_DAYS="${RETENTION_DAYS:-7}"

# Сжатие: gzip | zstd
COMPRESSOR="${COMPRESSOR:-gzip}"       # gzip по умолчанию
ZSTD_LEVEL="${ZSTD_LEVEL:-6}"          # если используешь zstd

# Хэши (sha256) для контроля целостности
WRITE_HASHES="${WRITE_HASHES:-1}"      # 1=включить, 0=выключить

# S3 (опционально): если указать, архив отправится на S3
S3_BUCKET="${S3_BUCKET:-}"             # напр.: s3://ai-trader-backups
S3_PREFIX="${S3_PREFIX:-}"             # напр.: prod/
AWS_CLI="${AWS_CLI:-aws}"              # путь к aws CLI

# Лог скрипта
BACKUP_LOG="${BACKUP_LOG:-$BACKUP_DIR/backup.log}"

# Лок-файл (защита от параллельных запусков)
LOCK_FILE="${LOCK_FILE:-$BACKUP_DIR/.backup.lock}"

# Часовой пояс в именах файлов — по умолчанию UTC, можно задать TZ в .env
DATE_FMT="${DATE_FMT:-%Y%m%d-%H%M%S}"

# ──────────────────────────────────────────────────────────────────────────────
# Подготовка окружения
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
touch "$BACKUP_LOG"

# Подхват .env (если есть)
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────
log() {
  local ts
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "$ts | $*" | tee -a "$BACKUP_LOG"
}

fail() {
  log "ERROR: $*"
  exit 1
}

compress_file() {
  local f="$1"
  if [[ "$COMPRESSOR" == "zstd" ]]; then
    command -v zstd >/dev/null 2>&1 || fail "zstd не установлен"
    zstd -$ZSTD_LEVEL -q --rm "$f"   # заменяет исходник .zst файлом
    echo "${f}.zst"
  else
    # gzip
    gzip -f -9 "$f"
    echo "${f}.gz"
  fi
}

write_hash() {
  local f="$1"
  if [[ "$WRITE_HASHES" == "1" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$(basename "$f")" > "${f}.sha256"
    elif command -v shasum >/dev/null 2>&1; then
      (cd "$(dirname "$f")" && shasum -a 256 "$(basename "$f")" > "$(basename "$f").sha256")
    fi
  fi
}

upload_s3() {
  local f="$1"
  [[ -n "$S3_BUCKET" ]] || return 0
  command -v "$AWS_CLI" >/dev/null 2>&1 || fail "aws cli не найден, отключи S3 или установи AWS CLI"
  local key="${S3_BUCKET%/}/"
  if [[ -n "$S3_PREFIX" ]]; then
    key+="${S3_PREFIX%/}/"
  fi
  key+=$(basename "$f")
  log "S3 upload → $key"
  "$AWS_CLI" s3 cp "$f" "$key" --only-show-errors
  if [[ "$WRITE_HASHES" == "1" && -f "${f}.sha256" ]]; then
    "$AWS_CLI" s3 cp "${f}.sha256" "${key}.sha256" --only-show-errors
  fi
}

rotate_local() {
  find "$BACKUP_DIR" -type f -name "*.tar.*" -mtime +"$RETENTION_DAYS" -print -delete | sed 's/^/rotated: /' || true
  find "$BACKUP_DIR" -type f -name "*.dump.*" -mtime +"$RETENTION_DAYS" -print -delete | sed 's/^/rotated: /' || true
  find "$BACKUP_DIR" -type f -name "*.db.*"   -mtime +"$RETENTION_DAYS" -print -delete | sed 's/^/rotated: /' || true
  find "$BACKUP_DIR" -type f -name "*.sha256" -mtime +"$RETENTION_DAYS" -print -delete | sed 's/^/rotated: /' || true
}

# Проверка архива (быстрая) — tar -tf
verify_tar() {
  local f="$1"
  if ! tar -tf "$f" >/dev/null 2>&1; then
    fail "Повреждённый архив: $f"
  fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Основная логика
# ──────────────────────────────────────────────────────────────────────────────
main() {
  umask 027
  local stamp
  stamp="$(date -u +"$DATE_FMT")"

  log "=== BACKUP START (type=$DB_TYPE, stamp=$stamp) ==="

  # Лок (не даём запустить второй параллельный процесс)
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    fail "Уже выполняется бэкап (lock: $LOCK_FILE)"
  fi

  local tmpdir="$BACKUP_DIR/_tmp_$stamp"
  mkdir -p "$tmpdir"

  # 1) Бэкап БД
  local db_out=""
  if [[ "$DB_TYPE" == "postgres" ]]; then
    command -v pg_dump >/dev/null 2>&1 || fail "pg_dump не найден"
    local dump_file="$tmpdir/${PGDATABASE}_${stamp}.dump"
    log "Postgres dump → $dump_file"
    # -Fc (custom), без owner/privileges для проще восстановления
    PGPASSWORD="${PGPASSWORD:-}" pg_dump \
      --host="$PGHOST" --port="$PGPORT" \
      --username="$PGUSER" \
      --format=custom \
      --no-owner --no-privileges \
      --file="$dump_file" \
      "$PGDATABASE"
    db_out="$(compress_file "$dump_file")"
    write_hash "$db_out"

  elif [[ "$DB_TYPE" == "sqlite" ]]; then
    command -v sqlite3 >/dev/null 2>&1 || fail "sqlite3 не найден"
    [[ -f "$SQLITE_PATH" ]] || fail "SQLite файл не найден: $SQLITE_PATH"
    local base
    base="$(basename "$SQLITE_PATH")"
    local copy_path="$tmpdir/${base}.${stamp}.db"
    log "SQLite backup → $copy_path"
    # Надёжный бэкап через sqlite3 .backup (без копирования "на горячую")
    sqlite3 "$SQLITE_PATH" ".backup '$copy_path'"
    db_out="$(compress_file "$copy_path")"
    write_hash "$db_out"
  else
    fail "Неизвестный DB_TYPE: $DB_TYPE (поддержка: postgres|sqlite)"
  fi

  # 2) Архив логов (опционально)
  local logs_out=""
  if [[ -d "$LOGS_DIR" ]]; then
    local tar_name="$tmpdir/logs_${stamp}.tar"
    log "Tar logs → $tar_name"
    tar -C "$LOGS_DIR" -cf "$tar_name" .
    logs_out="$(compress_file "$tar_name")"
    verify_tar "$logs_out"
    write_hash "$logs_out"
  else
    log "Каталог логов не найден (пропуск): $LOGS_DIR"
  fi

  # 3) Перенос из tmp в итоговую папку
  for f in "$tmpdir"/*; do
    [[ -e "$f" ]] || continue
    mv "$f" "$BACKUP_DIR/"
  done
  rmdir "$tmpdir" || true

  # 4) S3 (опционально)
  if [[ -n "$S3_BUCKET" ]]; then
    [[ -n "$db_out"    ]] && upload_s3 "$BACKUP_DIR/$(basename "$db_out")"
    [[ -n "$logs_out"  ]] && upload_s3 "$BACKUP_DIR/$(basename "$logs_out")"
  fi

  # 5) Ротация локальных бэкапов
  rotate_local

  log "=== BACKUP DONE (type=$DB_TYPE, stamp=$stamp) ==="
}

main "$@"
