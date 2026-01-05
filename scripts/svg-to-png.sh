#!/bin/bash

# Optional script to convert SVGs to PNGs if needed
# Requires: rsvg-convert or inkscape or ImageMagick

cd /Users/pgedge/pge/neurondb2 || exit 1

# Check for available tools
if command -v rsvg-convert &> /dev/null; then
    CONVERTER="rsvg-convert"
elif command -v inkscape &> /dev/null; then
    CONVERTER="inkscape"
elif command -v convert &> /dev/null; then
    CONVERTER="imagemagick"
else
    echo "No SVG converter found. Install rsvg-convert, inkscape, or ImageMagick."
    echo "On macOS: brew install librsvg"
    exit 1
fi

# Find all SVGs in blog/assets
find blog/assets -name "*.svg" -type f | while read -r svg_file; do
    png_file="${svg_file%.svg}.png"
    
    if [ ! -f "$png_file" ] || [ "$svg_file" -nt "$png_file" ]; then
        echo "Converting: $svg_file -> $png_file"
        
        case $CONVERTER in
            rsvg-convert)
                rsvg-convert -w 1200 -h 800 "$svg_file" > "$png_file"
                ;;
            inkscape)
                inkscape "$svg_file" --export-filename="$png_file" --export-width=1200 --export-height=800
                ;;
            imagemagick)
                convert -background none -resize 1200x800 "$svg_file" "$png_file"
                ;;
        esac
    fi
done

echo "Done converting SVGs to PNGs."

