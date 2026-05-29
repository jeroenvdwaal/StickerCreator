#!/usr/bin/env bash
# Build the Sticker Creator Flatpak.
#
# Usage:
#   packaging/flatpak/build.sh              # build only
#   INSTALL=1 packaging/flatpak/build.sh    # build + install into user remote
#   BUNDLE=1  packaging/flatpak/build.sh    # build + export single-file .flatpak
set -euo pipefail

cd "$(dirname "$0")/../.."

APP_ID=io.github.jeroenvdwaal.StickerCreator
MANIFEST=packaging/flatpak/${APP_ID}.yml
STATE=packaging/flatpak/.flatpak-builder
BUILDDIR=packaging/flatpak/build-dir
REPO=packaging/flatpak/repo

KDE_BRANCH=6.10
RUNTIMES=(
  "org.kde.Platform//${KDE_BRANCH}"
  "org.kde.Sdk//${KDE_BRANCH}"
  "io.qt.PySide.BaseApp//${KDE_BRANCH}"
)

if ! command -v flatpak-builder >/dev/null; then
  echo "error: flatpak-builder not installed (dnf install flatpak-builder)" >&2
  exit 1
fi

flatpak --user remote-add --if-not-exists flathub \
  https://flathub.org/repo/flathub.flatpakrepo

flatpak install --user -y --noninteractive flathub "${RUNTIMES[@]}"

INSTALL_FLAG=""
[[ "${INSTALL:-}" == "1" ]] && INSTALL_FLAG="--install"

REPO_FLAG=""
[[ "${BUNDLE:-}" == "1" ]] && REPO_FLAG="--repo=${REPO}"

flatpak-builder \
  --force-clean --user \
  --install-deps-from=flathub \
  --state-dir="${STATE}" \
  ${INSTALL_FLAG} ${REPO_FLAG} \
  "${BUILDDIR}" "${MANIFEST}"

if [[ "${BUNDLE:-}" == "1" ]]; then
  OUT="packaging/flatpak/${APP_ID}.flatpak"
  flatpak build-bundle "${REPO}" "${OUT}" "${APP_ID}"
  echo "bundle: ${OUT}"
fi
