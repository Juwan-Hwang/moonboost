#!/bin/sh
# Bundle the JS build for the browser demo in docs/ (served by GitHub Pages).
set -e
moon build --target js --release
npx -y esbuild@0.24 _build/js/release/build/web/web.js --minify --format=esm \
  --outfile=docs/moonboost.min.js
