# Record a Watch Dashboard GIF

This guide shows two ways to record `pulse watch` as a short GIF for the README.

## Option A: Asciinema + agg (recommended)

1) Install tools:

- macOS (Homebrew):

```bash
brew install asciinema ffmpeg
pipx install agg  # or: pip install agg
```

- Linux (Debian/Ubuntu):

```bash
sudo apt-get install asciinema ffmpeg
pipx install agg  # or: pip install agg --user
```

2) Record a short session (10–15s is plenty):

```bash
asciinema rec -c "pulse watch --interval 2" demo.cast
# press Ctrl-C after ~10s to stop
```

3) Render to GIF with agg (crisp text):

```bash
agg --font-size 14 \
    --no-loop \
    --speed 1.0 \
    --theme dracula \
    demo.cast assets/watch-demo.gif
```

Tip: If the path has spaces, quote it or run from repo root.

## Option B: Terminalizer

1) Install:

```bash
npm install -g terminalizer
```

2) Record:

```bash
terminalizer record watch-demo
# run: pulse watch --interval 2
# stop recording with Ctrl-D
```

3) Render to GIF:

```bash
terminalizer render watch-demo.yml -o assets/watch-demo.gif
```

## Commit the demo

```bash
git add assets/watch-demo.gif
git commit -m "docs: add watch dashboard GIF"
git push origin main
# optional: push to upstream as well
git push upstream main
```

If you prefer, send me the `demo.cast` or the rendered `watch-demo.gif` and I’ll commit it for you.
