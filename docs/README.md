# Website

The HydroGym documentation is built with [Docusaurus](https://docusaurus.io/) (Node.js ≥ 20, see `.node-version`).

## Installation

```bash
npm ci
```

## Local development

```bash
npm start
```

This starts a local development server and opens a browser window; most changes are reflected live.

## Build

```bash
npm run build
```

This regenerates the API pages from the docstrings (`npm run generate-api`, needs `uv`) and writes the static site
into `build/`. The site is deployed to GitHub Pages by `.github/workflows/deploy.yml`.
