#!/bin/bash

output_file="all_codes.txt"
> "$output_file"  # Clear or create the output file

# Find all .py files, excluding any in folders with 'model' in the path
find . -type f -name "*.py" ! -path "*/model/*" ! -path "*/model" | while read -r file; do
  echo "=============== $file ===============" >> "$output_file"
  cat "$file" >> "$output_file"
  echo -e "\n\n" >> "$output_file"
done

echo "✅ All Python code (excluding 'model' folders) collected in $output_file"
