set -ex

DOCS_PATH="$(dirname -- "${BASH_SOURCE[0]}")"            # relative
DOCS_PATH="$(cd -- "$DOCS_PATH" && pwd)"    # absolutized and normalized
if [[ -z "$DOCS_PATH" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi

cd $DOCS_PATH
rm -rf _build/html
make clean html

TARGET_PATH=/tmp/t3w_ghpages
rm -rf $TARGET_PATH
git clone git@github.com:tjyuyao/t3w.git $TARGET_PATH
cd $TARGET_PATH
git checkout gh-pages
rm -r *
cp -r $DOCS_PATH/_build/html/* .
git add .
git commit -m "update docs on `date`"
git push
