#!/bin/bash

echo "Removing emojis from project..."

# Generate emoji list if missing
if [ ! -f emoji-list.txt ]; then
  echo "emoji-list.txt not found. Run find-emojis.sh first."
  exit 1
fi

# Read emojis into a sed-safe pattern
EMOJI_PATTERN=$(sed 's/[]\/$*.^|[]/\\&/g' emoji-list.txt | tr '\n' '|')
EMOJI_PATTERN="(${EMOJI_PATTERN%|})"

# Run through files
find . \
  -path "./.git" -prune -o \
  -path "./node_modules" -prune -o \
  -path "./venv" -prune -o \
  -path "./env" -prune -o \
  -type f -exec gsed -i "s/$EMOJI_PATTERN/ /g" {} +

echo "Done. Emojis removed."
