//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#define NR 2
#define KR 1
static const size_t kai_num_bytes_input = sizeof(uint32_t);
static const size_t kai_num_bytes_output = sizeof(uint32_t);
static const size_t kai_num_bytes_bias = sizeof(uint32_t);

size_t kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(void) {
    return NR * kai_get_sme_vector_length_u32() / KR;
}

size_t kai_get_rhs_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

static size_t kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme() *
        (kai_num_bytes_bias + k_chunk_count * kai_roundup(k_chunk_length, KR) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme();
    return block_idx *
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(k_chunk_count, k_chunk_length);
}

size_t kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_rounded_up = kai_roundup(n, kai_get_n_step_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme());
    return kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
        n_rounded_up, k_chunk_count, k_chunk_length);
}

void kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_row_stride, const void* rhs, const void* bias,
    void* rhs_packed) {
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);

    size_t height = k_chunk_length;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_row_stride;

    size_t out_stride =
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(k_chunk_count, k_chunk_length);
    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x22, %x[out]\n"
        "mov x21, %x[width]\n"
        "ptrue p2.b\n"
        "1:"  // Bias: Full loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x21, #0x0\n"
        "ld1w { z17.s }, p1/Z, [%x[bias]]\n"
        "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
        "incb %x[bias], ALL, MUL #2\n"
        "st1w { z17.s }, p2, [x22]\n"
        "st1w { z16.s }, p2, [x22, #1, MUL VL]\n"
        "add x22, x22, %x[out_stride]\n"
        "bgt 1b\n"
        "incb %x[out], ALL, MUL #2\n"
        "mov x28, %x[k_chunk_count]\n"
        "2:"  // Chunk Loop
        "mov x27, %x[height]\n"
        "cmp x27, #0x4\n"
        "blt 6f\n"
        "3:"  // Main row loop: Head
        "mov x26, %x[in]\n"
        "mov x25, %x[out]\n"
        "add x24, x26, %x[in_stride]\n"
        "sub x27, x27, #0x4\n"
        "add x23, x24, %x[in_stride]\n"
        "mov x22, %x[width]\n"
        "add x21, x23, %x[in_stride]\n"
        "add %x[in], x21, %x[in_stride]\n"
        "4:"  // Main row loop: Column loop
        "mov x20, x22\n"
        "decw x22, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x22, #0x0\n"
        "ld1w { z23.s }, p1/Z, [x26]\n"
        "ld1w { z22.s }, p0/Z, [x26, #1, MUL VL]\n"
        "addvl x26, x26, #2\n"
        "ld1w { z21.s }, p1/Z, [x24]\n"
        "ld1w { z20.s }, p0/Z, [x24, #1, MUL VL]\n"
        "addvl x24, x24, #2\n"
        "ld1w { z19.s }, p1/Z, [x23]\n"
        "ld1w { z18.s }, p0/Z, [x23, #1, MUL VL]\n"
        "addvl x23, x23, #2\n"
        "ld1w { z17.s }, p1/Z, [x21]\n"
        "ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        "st1w { z23.s }, p2, [x25]\n"
        "st1w { z22.s }, p2, [x25, #1, MUL VL]\n"
        "st1w { z21.s }, p2, [x25, #2, MUL VL]\n"
        "st1w { z20.s }, p2, [x25, #3, MUL VL]\n"
        "st1w { z19.s }, p2, [x25, #4, MUL VL]\n"
        "st1w { z18.s }, p2, [x25, #5, MUL VL]\n"
        "st1w { z17.s }, p2, [x25, #6, MUL VL]\n"
        "st1w { z16.s }, p2, [x25, #7, MUL VL]\n"
        "add x25, x25, %x[out_stride]\n"
        "bgt 4b\n"
        "cmp x27, #0x4\n"
        "addvl %x[out], %x[out], #8\n"
        "bge 3b\n"
        "cbz x27, 10f\n"
        "6:"  // Main loop skip
        "7:"  // Tail row loop: Head
        "mov x26, %x[in]\n"
        "cntw x22, ALL, MUL #8\n"
        "add %x[in], x26, %x[in_stride]\n"
        "mov x25, %x[out]\n"
        "sub x27, x27, #0x1\n"
        "mov x21, %x[width]\n"
        "8:"  // Tail row loop: Column loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x21, #0x0\n"
        "ld1w { z17.s }, p1/Z, [x26]\n"
        "ld1w { z16.s }, p0/Z, [x26, #1, MUL VL]\n"
        "add x26, x26, x22\n"
        "st1w { z17.s }, p2, [x25]\n"
        "st1w { z16.s }, p2, [x25, #1, MUL VL]\n"
        "add x25, x25, %x[out_stride]\n"
        "bgt 8b\n"
        "cmp x27, #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 7b\n"
        "10:"  // Done
        "sub x28, x28, #0x1\n"
        "cbnz x28, 2b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [in] "+&r"(in), [out] "+&r"(out)
        : [height] "r"(height), [in_stride] "r"(in_stride), [k_chunk_count] "r"(k_chunk_count),
          [out_stride] "r"(out_stride), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z10", "z11", "z12",
          "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
          "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
