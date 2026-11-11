"""Write a copy of flashinfer's generated MoE include root whose flashinferMetaInfo.h keeps only the
trtllm-gen kernel configs of the FP4 MoE GEMMs: NVFP4 (e2m1 x e2m1, the target's experts) and MXFP4
weights with MXFP8 activations (MxE2m1 x MxE4m3, the MTP draft's experts).

    python filter_metainfo.py /dsv4/deps/moe_build/gen_root /dsv4/deps/moe_build/gen_root_fp4

gcc 13.3 segfaults compiling trtllm_batched_gemm_runner.cu against the full 6966-entry config list.
The native port selects its MoE tactic by kernel name, so dropping unrelated configs (BF16/FP8 GEMMs,
other dtypes) changes no kept entry; only the indices of the kept ones shift.
"""
import re
import shutil
import sys
from pathlib import Path

KEEP = re.compile(r'"(bmm_E2m1_E2m1E2m1_Fp32_[^"]*|bmm_Bfloat16_E2m1E2m1_Fp32_[^"]*'
                  r'|bmm_MxE4m3_MxE2m1MxE4m3_Fp32_[^"]*|bmm_Bfloat16_MxE2m1MxE4m3_Fp32_[^"]*)"')


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    lines = (src / 'flashinferMetaInfo.h').read_text().splitlines(keepends=True)
    out, entry, kept, total = [], None, 0, 0

    def flush():
        nonlocal entry, kept, total
        if entry is not None:
            total += 1
            if KEEP.search(entry[0]):
                out.extend(entry)
                kept += 1
        entry = None

    for line in lines:
        if line.startswith('{nullptr, 0, '):
            flush()
            entry = [line]
        elif entry is not None and (line.startswith('#') or line.startswith('};')):
            flush()
            out.append(line)
        elif entry is not None:
            entry.append(line)
        else:
            out.append(line)
    flush()
    text = re.sub(r'tllmGenBatchedGemmListLen = \d+;', f'tllmGenBatchedGemmListLen = {kept};', ''.join(out))
    (dst / 'flashinferMetaInfo.h').write_text(text)
    print(f'kept {kept} of {total} configs')


if __name__ == '__main__':
    main()
