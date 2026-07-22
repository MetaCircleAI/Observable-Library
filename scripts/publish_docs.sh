#!/usr/bin/env bash
set -euo pipefail

artifact_input=${1:-}
test -n "$artifact_input"
test "${DOCS_TARGET_REPOSITORY:-}" = "MetaCircleAI/Observable-Library"
test "${DOCS_TARGET_BRANCH:-}" = "gh-pages"
test -n "${DOCS_DEPLOY_TOKEN:-}"
test -n "${DOCS_GIT_NAME:-}"
test -n "${DOCS_GIT_EMAIL:-}"
command -v git >/dev/null
command -v rsync >/dev/null

artifact_dir=$(cd "$artifact_input" && pwd -P)
test "$artifact_dir" != "/"
test -f "$artifact_dir/index.html"
test -f "$artifact_dir/zh/index.html"
test -f "$artifact_dir/404.html"
test -f "$artifact_dir/.nojekyll"

publish_root=$(mktemp -d "${TMPDIR:-/tmp}/observable-docs-publish.XXXXXX")
publish_dir="$publish_root/site"
askpass="$publish_root/git-askpass.sh"

cleanup() {
  if test -n "${publish_root:-}" && test "$publish_root" != "/"; then
    rm -rf -- "$publish_root"
  fi
}
trap cleanup EXIT

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "$1" in' \
  '  *Username*) printf "%s\\n" "$GIT_USERNAME" ;;' \
  '  *Password*) printf "%s\\n" "$GIT_PASSWORD" ;;' \
  '  *) exit 1 ;;' \
  'esac' > "$askpass"
chmod 700 "$askpass"

export GIT_ASKPASS="$askpass"
export GIT_TERMINAL_PROMPT=0
export GIT_USERNAME=x-access-token
export GIT_PASSWORD="$DOCS_DEPLOY_TOKEN"

remote_url="https://github.com/${DOCS_TARGET_REPOSITORY}.git"
if git ls-remote --exit-code --heads "$remote_url" "refs/heads/${DOCS_TARGET_BRANCH}" >/dev/null 2>&1; then
  git clone --depth 1 --branch "$DOCS_TARGET_BRANCH" "$remote_url" "$publish_dir"
else
  mkdir "$publish_dir"
  git -C "$publish_dir" init
  git -C "$publish_dir" remote add origin "$remote_url"
  git -C "$publish_dir" switch --orphan "$DOCS_TARGET_BRANCH"
fi

rsync -a --delete --exclude ".git/" "$artifact_dir/" "$publish_dir/"
git -C "$publish_dir" config user.name "$DOCS_GIT_NAME"
git -C "$publish_dir" config user.email "$DOCS_GIT_EMAIL"
git -C "$publish_dir" add --all

if git -C "$publish_dir" diff --cached --quiet; then
  echo "Documentation artifact is already published."
  exit 0
fi

git -C "$publish_dir" commit -m "Publish documentation for ${GITHUB_SHA:-manual run}"
git -C "$publish_dir" push origin "HEAD:${DOCS_TARGET_BRANCH}"
