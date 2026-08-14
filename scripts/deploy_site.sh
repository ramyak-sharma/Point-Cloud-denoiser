#!/usr/bin/env bash
# Publish site/ to the public showcase repo that GitHub Pages serves.
#
# The research repo is private; GitHub Pages only serves private repos on paid
# plans, and the site has to be publicly reachable to be shareable by link
# anyway. So the built site lives in its own public repo, and the model code,
# checkpoints and raw data stay private here.
set -euo pipefail
REPO="${1:-lidar-denoising-showcase}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"

cp -R "$ROOT/site/." "$STAGE/"
cp "$ROOT/report/final_report.pdf" "$ROOT/report/viva_qna.pdf" "$STAGE/assets/" 2>/dev/null || true
touch "$STAGE/.nojekyll"   # keep Pages from eating files that start with _

cd "$STAGE"
git init -q -b main
git add -A
git -c user.email="noreply@github.com" -c user.name="capstone-deploy" \
    commit -q -m "Deploy showcase site ($(date -u +%Y-%m-%dT%H:%MZ))"

if ! gh repo view "$REPO" >/dev/null 2>&1; then
  gh repo create "$REPO" --public \
    --description "Interactive showcase — LiDAR point cloud denoising in adverse weather (Capstone CPG No. 5, TIET)"
fi
git remote add origin "$(gh repo view "$REPO" --json sshUrl -q .sshUrl)"
git push -qf origin main
gh api -X POST "repos/$(gh api user -q .login)/$REPO/pages" \
   -f 'source[branch]=main' -f 'source[path]=/' >/dev/null 2>&1 || true

echo "deployed: https://$(gh api user -q .login).github.io/$REPO/"
rm -rf "$STAGE"
