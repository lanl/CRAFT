#!/bin/bash

# Define the source and destination directories
SRC_DIR="."
DEST_DIR="$HOME/CRAFT_installation"

# Create the destination directory
mkdir -p "$DEST_DIR"

# Copy files and directories, excluding 'dep' directory
for item in "$SRC_DIR"/*; do
    if [ "$(basename "$item")" != "dep" ] && [ "$(basename "$item")" != "install.sh" ]; then
        cp -R "$item" "$DEST_DIR"
    fi
done

echo "Installation completed. Files are copied to $DEST_DIR"
