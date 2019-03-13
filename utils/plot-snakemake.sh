TMPF="$(mktemp).png" bash -c 'snakemake --dag | dot -Tpng > $TMPF && eog $TMPF'
