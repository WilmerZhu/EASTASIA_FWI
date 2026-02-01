#!/bin/bash
cd "$(dirname "$0")"
mkdir -p output

for file in *.mmd; do
    if [ -f "$file" ]; then
        echo "Converting $file..."
        npx -y @mermaid-js/mermaid-cli@latest \
            -i "$file" \
            -o "output/${file%.mmd}.svg" \
            --backgroundColor transparent \
            --configFile /dev/null 2>&1 | grep -v "npm warn" || true
    fi
done

echo "Done! Converted $(ls output/*.svg 2>/dev/null | wc -l) files."
