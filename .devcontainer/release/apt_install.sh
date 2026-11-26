#!/bin/sh
# apt-get install the given packages (no recommends) and clean up. With
# UBUNTU_MIRROR set (e.g. http://<country>.archive.ubuntu.com/ubuntu/), the Ubuntu
# archive and security URIs point at that mirror for this install only, for
# build hosts that cannot reach archive.ubuntu.com; the image keeps the
# default sources.
set -eu
src=/etc/apt/sources.list.d/ubuntu.sources
if [ -n "${UBUNTU_MIRROR:-}" ] && [ -f "${src}" ]; then
    cp "${src}" /tmp/ubuntu.sources.orig
    sed -i -e "s|http://archive.ubuntu.com/ubuntu/*|${UBUNTU_MIRROR}|g" \
           -e "s|http://security.ubuntu.com/ubuntu/*|${UBUNTU_MIRROR}|g" "${src}"
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends "$@"
rm -rf /var/lib/apt/lists/*
if [ -f /tmp/ubuntu.sources.orig ]; then mv /tmp/ubuntu.sources.orig "${src}"; fi
