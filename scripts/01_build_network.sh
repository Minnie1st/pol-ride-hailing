#!/bin/bash
cd ../tartu

netconvert \
    --osm-files tartu.osm \
    -o tartu.net.xml \
    --geometry.remove \
    --ramps.guess \
    --junctions.join \
    --tls.guess-signals \
    --tls.discard-simple \
    --tls.join

echo "Done. Check tartu/tartu.net.xml"