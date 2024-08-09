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

# Initialize a variable to track existing files
existing_files=""

# Check if any of the PNG files already exist
if [ -f "$output_png_16" ]; then
    existing_files="$existing_files $output_png_16"
fi
if [ -f "$output_png_32" ]; then
    existing_files="$existing_files $output_png_32"
fi
if [ -f "$output_png_48" ]; then
    existing_files="$existing_files $output_png_48"
fi

# If any files exist, output the list and exit
if [ -n "$existing_files" ]; then
    echo "The following PNG files already exist:$existing_files"
    echo "Please ensure they do not exist."
    exit 1
fi

# Create resized PNG files
convert "$input_image" -resize 16x16 "$output_png_16"
convert "$input_image" -resize 32x32 "$output_png_32"
convert "$input_image" -resize 48x48 "$output_png_48"

# Create the favicon.ico file
convert "$output_png_16" "$output_png_32" "$output_png_48" "$output_ico"

# Delete the temporary PNG files
rm "$output_png_16" "$output_png_32" "$output_png_48"

echo "Favicon and intermediate files created in $output_dir"
