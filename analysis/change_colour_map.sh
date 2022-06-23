#!/bin/sh

sed -i.old 's/plots1/squarecolourmap/g' */_/1/*{FF,SF,SR}.tikz
sed -i.old 's/plots1/cubecolourmap/g' */*/*/*AllProbes.tikz
sed -i.old 's/plots1/slicecolourmap/g' */*/*/*Slice_*.tikz
sed -i.old 's/plots1/maxcolourmap/g' max_*/*.tikz
sed -i.old 's/plots1/diffcolourmap/g' diff_*/*.tikz