#!/usr/bin/env bash
# Компилира дипломната работа с Tectonic (XeTeX, Unicode/кирилица).
# Употреба: ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic не е намерен. Инсталирай с: brew install tectonic" >&2
  exit 1
fi

# Tectonic изисква изходната папка да съществува предварително.
mkdir -p build

# --keep-intermediates + --synctex за удобство; Tectonic сам решава
# колко пъти да компилира и тегли нужните пакети.
tectonic -X compile main.tex --keep-intermediates --synctex --outdir build
echo "Готово: build/main.pdf"
