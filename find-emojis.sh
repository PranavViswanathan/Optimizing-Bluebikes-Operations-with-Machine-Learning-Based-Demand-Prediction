#!/bin/bash

# Directories you want to ignore
EXCLUDES="--exclude-dir=.git --exclude-dir=node_modules --exclude-dir=venv --exclude-dir=env"

# Emoji regex ranges (Unicode)
EMOJI_REGEX="[\x{1F300}-\x{1FAFF}\x{1F000}-\x{1F6FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]"

echo "Scanning for emojis..."
ggrep -RIn -P $EXCLUDES "$EMOJI_REGEX" .
