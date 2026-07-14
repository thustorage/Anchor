#!/bin/bash

# Source directory containing *.cpp files
source_directory="$PWD"

# Check if the source directory exists
if [ ! -d "$source_directory" ]; then
    echo "Source directory not found: $source_directory"
    exit 1
fi

# Find all *.cpp files in the source directory and its subdirectories
cpp_files=$(find "$source_directory" -type f -name "*.hip")

for file in $cpp_files; do
    if [ -e "$file" ]; then
        new_name="${file%.hip}.cpp"
        mv "$file" "$new_name"
        echo "Renamed: $file to $new_name"
    fi
done

echo "File renaming complete."
