#!/usr/bin/env bash
# Run from repo root:  bash data/fetch_data.sh
set -e
mkdir -p data/raw && cd data/raw
[ -d InjecAgent ] || git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git
[ -d BIPIA ] || git clone --depth 1 https://github.com/microsoft/BIPIA.git
echo "Done. Full data in data/raw/{InjecAgent,BIPIA}"
