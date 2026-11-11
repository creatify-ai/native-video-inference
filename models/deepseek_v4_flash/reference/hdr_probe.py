"""Probe: declarations of the runtime's internal launch entry points in the CUDA headers."""
import subprocess
print(subprocess.run(['bash', '-c', 'sed -n 2920,2970p /usr/local/cuda/include/crt/device_functions.h; sed -n 120,150p /usr/local/cuda/include/crt/host_runtime.h'], capture_output=True, text=True).stdout)
