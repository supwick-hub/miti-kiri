#!/bin/zsh
# Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
# The 1.0 model's outputs must not change: four runs on a deterministic test mix are hashed and compared with
# tests/golden.sha256. Core ML's GPU output is machine-specific, so the hashes hold on the Mac that recorded
# them (see tests/golden.sha256's first line); another machine records its own.
#
#   tools/golden.sh record [BIN]   # (re)write the hashes; only from a known-good binary
#   tools/golden.sh check  [BIN]   # exit 1 on any difference
set -u
mode=$1; here=${0:A:h}; root=${here:h}
BIN=${2:-$root/target/release/miti-kiri}
M=$root/model/miti-kiri-1.0.mlmodelc
F=$root/model/miti-kiri-1.0-fold.mlmodelc
O=${GOLDEN_OUT:-${TMPDIR:-/tmp}/miti-kiri-golden}
G=$root/tests/golden.sha256
rm -rf $O; mkdir -p $O
python3 $here/make_test_mix.py $O/mix.wav
run() { local tag=$1; shift; nice -n 10 $BIN $O/mix.wav -o $O/$tag "$@" 2>$O/$tag.log || { echo "FAIL $tag: $(tail -1 $O/$tag.log)"; exit 1; }; }
run default -m $M
run o0-none -m $M --overlap 0 --consist none
run o50-pcm16 -m $M --overlap 0.5 --pcm16
run fold-default -m $F
(cd $O && shasum -a 256 */*.wav | sort -k2) > $O/hashes
if [[ $mode == record ]]; then
  { echo "# $(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), Core ML GPU"; cat $O/hashes; } > $G
  echo "recorded $(wc -l < $O/hashes) hashes to $G"
else
  if diff <(grep -v '^#' $G) $O/hashes; then echo "miti-kiri 1.0 bit-identical: $(wc -l < $O/hashes) files"; else echo "miti-kiri 1.0 CHANGED"; exit 1; fi
fi
