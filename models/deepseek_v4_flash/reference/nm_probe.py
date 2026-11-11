"""Probe: launch-related undefined symbols of the native build's CUDA objects."""
import glob, subprocess
for o in sorted(glob.glob('/build/target/release/build/dsv4-flash-native-*/out/moe_objects/*.o'))[:40]:
    syms = subprocess.run(['nm', '-u', o], capture_output=True, text=True).stdout.split()
    hits = sorted({s for s in syms if 'aunch' in s or 'cudaLaunch' in s})
    print(o.split('/')[-1], hits)
