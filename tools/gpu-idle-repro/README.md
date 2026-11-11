# Wrong results after the GPU has been idle (RADV, Ryzen 7000 integrated GPU)

`repro.c` is plain Vulkan 1.1 with no library code. Each iteration it uploads a 786 KB float buffer with a transfer copy, waits for the queue
to go idle, sleeps, runs one compute dispatch (a minimum over the middle axis of a 2 x 12 x 128 x 64 tensor, one 256-thread workgroup per
output, a shared-memory tree), copies the 1536 results to host memory and compares them with a CPU reference, bit for bit. Every barrier the
Vulkan specification asks for is there; the shader (`ReduceBlock.spv`, compiled from `RecursiveAI.OneGpu.Kernels`' `Reduce.hlsl`) has a
barrier after every shared-memory step.

```sh
# Vulkan-Headers (include/vulkan, include/vk_video) on the include path
gcc -O1 -I<headers> -o repro repro.c -l:libvulkan.so.1
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json PUSH=push.bin ENTRY=ReduceBlock NGROUPS=1536 \
    ./repro <iterations> <sleep ms> <warm 0|1> ReduceBlock.spv
```

Observed on a Ryzen 9 7950X3D integrated GPU (RADV RAPHAEL_MENDOCINO), 1000 iterations, 20 ms sleep:

| Stack | Wrong dispatches |
|---|---|
| Mesa 26.0.3, kernel 7.0.0-42, newer GFX/SMU firmware (mec 0x1a, pfp 0x12, smc 0x00544fdf) | 22 to 74 per 1000, in every run |
| the same with `RADV_DEBUG=llvm` (LLVM compiler backend instead of ACO) | 61 |
| Mesa 25.2.8, kernel 7.0.0-38, older firmware (mec 0x14, pfp 0x0e, smc 0x04540400), same GPU model | 0 in 3000 |
| NVIDIA RTX 4070 | 0 |

The failure depends on the GPU clock, not on the data or on synchronisation:

- with no sleep between the upload and the dispatch: 0 in 1000; with 2 ms: 16 in 1000; with 20 ms: about 3%;
- with another process keeping the GPU busy (gpu clock 2200 MHz instead of the idle 600 MHz): 0 in 2000, against 22 in 1000 without it;
- in the library's own test run (same symptom), a full `ALL_COMMANDS` memory barrier before every dispatch, or after it, changed nothing (25 and 40 wrong in 6000, against 27);
- the wrong value is one output of 1536 (a NaN with a few low mantissa bits set, or a different float), is stable when read again, and a repeat
  of the same dispatch is correct;
- a six-workgroup kernel without shared memory showed 0 in 4300; a one-workgroup-per-output shared-memory tree written in WGSL failed 2 in 3000
  and, with 64-thread workgroups, 0 in 3000, so the rate depends on the kernel and the cause is not pinned to shared memory.
