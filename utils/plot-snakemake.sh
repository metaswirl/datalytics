#!/bin/bash
if [ "x$1x" == "xx" ]; then
  IMGOUT="$(mktemp).png" 
else
  IMGOUT="$1"
fi
echo "snakemake -n --dag | dot -Tpng > \"$IMGOUT\""
snakemake -n --dag | dot -Tpng > "$IMGOUT"
echo "Opening $IMGOUT"
eog $IMGOUT
