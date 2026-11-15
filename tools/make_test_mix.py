#!/usr/bin/env python3
# Copyright (c) 2026 Supriya Wickrematillake. MIT licence; see LICENSE.
"""Writes a deterministic 30 s stereo 44.1 kHz 16-bit WAV (tones, a pulse train and LCG noise): the input
of tools/golden.sh, so the hashes can be reproduced without any music."""
import struct, sys, math

def main(path):
    n = 30 * 44100
    seed = 0x12345678
    out = bytearray()
    for i in range(n):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        noise = (seed >> 16) / 65536.0 - 0.5
        t = i / 44100.0
        pulse = 0.5 if (i % 22050) < 400 else 0.0
        l = 0.3 * math.sin(2 * math.pi * 55 * t) + 0.2 * math.sin(2 * math.pi * 440 * t) + pulse * math.sin(2 * math.pi * 90 * t) + 0.05 * noise
        r = 0.25 * math.cos(2 * math.pi * 110 * t) + 0.15 * math.sin(2 * math.pi * 1760 * t) + 0.05 * noise
        clip = lambda x: max(-32768, min(32767, int(x * 32767)))
        out += struct.pack('<hh', clip(l), clip(r))
    hdr = b'RIFF' + struct.pack('<I', 36 + len(out)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 2, 44100, 44100 * 4, 4, 16) + b'data' + struct.pack('<I', len(out))
    with open(path, 'wb') as f:
        f.write(hdr + out)

main(sys.argv[1])
