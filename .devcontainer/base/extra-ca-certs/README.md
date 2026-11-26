Extra CA certificates trusted inside the image (build time and run time).

Behind a TLS-intercepting proxy the downloads in Dockerfile.base (wget, curl,
uv) and later pip / Hugging Face downloads fail with certificate errors. Copy
the proxy's root CA here as a PEM file ending in `.crt` before building, e.g.

    cp /usr/local/share/ca-certificates/<corp-ca>.crt .devcontainer/base/extra-ca-certs/

`*.crt` files in this directory are git-ignored. With no files here the
step is a no-op.
