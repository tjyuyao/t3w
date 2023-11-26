# steps to update docs

1. in the `docs` dir, run `make clean html` to test html generation, clean reported issues.
2. in `_build/html` directory, run `python -m http.server` to preview it locally.
2. run `bash build.sh`, which will remake the docs, pull the gh-pages branch in a tmp directory, update the docs and push it back to the repo.
3. wait a few minute and refresh the page to see.
