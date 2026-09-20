#!/bin/zsh
# Start the ENCS5323 demo control panel, then open it in the browser.
cd "$(dirname "$0")/.."
PORT=8000
echo "Starting demo at http://127.0.0.1:$PORT  (Ctrl-C to stop)"
( sleep 2; open http://127.0.0.1:$PORT ) &
exec .venv/bin/python demo/app.py $PORT
