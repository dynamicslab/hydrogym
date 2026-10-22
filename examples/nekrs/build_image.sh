#!/usr/bin/env bash
# Build the HydroGym NekRS image from the committed state of a branch (default: the current one).
# Usage: examples/nekrs/build_image.sh
#        BRANCH=nekrs-backend TAG=localhost/hydrogym-nekrs:26.0-gpumpi-turing_ampere-hgnekrs examples/nekrs/build_image.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=${BASE:-localhost/hydrogym-nekrs:26.0-gpumpi-turing_ampere}
TAG=${TAG:-localhost/hydrogym-nekrs:26.0-gpumpi-turing_ampere-hgnekrs}
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "[build_image] uncommitted changes in $REPO are not included (the image takes the committed $BRANCH)"
fi
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
git -C "$REPO" bundle create "$CTX/hydrogym.bundle" "$BRANCH"
cp "$HERE/Containerfile" "$CTX/"
podman build --build-arg BASE="$BASE" --build-arg BRANCH="$BRANCH" -t "$TAG" "$CTX"
echo "[build_image] $TAG ($BRANCH $(git -C "$REPO" rev-parse --short "$BRANCH"))"
