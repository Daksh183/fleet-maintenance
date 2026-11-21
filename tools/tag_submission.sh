#!/usr/bin/env bash
# Record one Kaggle submission: annotated git tag (unique number + date) + a row in SUBMISSIONS.md.
# Usage (run right after uploading, from the repo root, with the code that produced the file committed):
#   tools/tag_submission.sh submissions/submission_v9_k4.csv notebooks/v9_cost_model_capacity.ipynb "v9: shape cost model + prior shift k=4"
set -euo pipefail
file="$1"; nbk="$2"; desc="${3:-}"; score="${4:-_pending_}"
[ -f "$file" ] || { echo "no such file: $file"; exit 1; }
[ -f "$nbk" ] || { echo "no such notebook: $nbk"; exit 1; }
git diff --quiet HEAD -- src notebooks tools || { echo "commit your code changes first"; exit 1; }
n=$(( $(git tag -l 'sub[0-9][0-9]-*' | wc -l) + 1 ))
num=$(printf "%02d" "$n"); date=$(date +%Y-%m-%d)
name=$(basename "$file" .csv); name=${name#submission_}
tag="sub${num}-${date}-${name}"
sha=$(shasum -a 256 "$file" | cut -c1-12)
git tag -a "$tag" -m "Kaggle submission #${num} on ${date}
file: ${file} (sha256 ${sha})
notebook: ${nbk}
${desc}"
printf '| %s | %s | `%s` | `%s` | `%s` | %s | %s |\n' "$num" "$date" "$tag" "$file" "$nbk" "$desc" "$score" >> SUBMISSIONS.md
git add SUBMISSIONS.md && git commit -q -m "Log Kaggle submission #${num} (${name})"
git push -q origin HEAD "$tag"
echo "tagged and pushed $tag"
