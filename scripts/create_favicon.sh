#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 -i <input_image_file_path> -o <output_dir_path>"
    exit 1
}

# Parse command line arguments
while getopts "i:o:" opt; do
    case "$opt" in
        i) input_image="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if input and output paths are provided
if [ -z "$input_image" ] || [ -z "$output_dir" ]; then
    usage
fi

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is not installed. Please install it first."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Define output file paths
output_png_16="$output_dir/favicon-16x16.png"
output_png_32="$output_dir/favicon-32x32.png"
output_png_48="$output_dir/favicon-48x48.png"
output_ico="$output_dir/favicon.ico"

# Create resized PNG files
convert "$input_image" -resize 16x16 "$output_png_16"
convert "$input_image" -resize 32x32 "$output_png_32"
convert "$input_image" -resize 48x48 "$output_png_48"

# Create the favicon.ico file
convert "$output_png_16" "$output_png_32" "$output_png_48" "$output_ico"

echo "Favicon and intermediate files created in $output_dir"
