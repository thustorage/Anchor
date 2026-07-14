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
#include "kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

#define NR 2
#define KR 2
static const size_t kai_num_bytes_input = sizeof(uint16_t);
static const size_t kai_num_bytes_output = sizeof(uint16_t);
static const size_t kai_num_bytes_bias = sizeof(uint16_t);

#define MAX_N_STEP (NR * ((KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(uint16_t)) / KR))

size_t kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(void) {
    return NR * kai_get_sme_vector_length_u16() / KR;
}

size_t kai_get_rhs_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

static size_t kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme() *
        (kai_num_bytes_bias + k_chunk_count * kai_roundup(k_chunk_length, KR) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme();
    return block_idx *
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(k_chunk_count, k_chunk_length);
}

size_t kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_rounded_up = kai_roundup(n, kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme());
    return kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
        n_rounded_up, k_chunk_count, k_chunk_length);
}

void kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
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

    KAI_ASSERT(kai_get_n_step_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme() <= MAX_N_STEP);
    uint16_t pad_row[MAX_N_STEP];
    if (height % KR) {
        memset(pad_row, 0, MAX_N_STEP * sizeof(uint16_t));
    }

    size_t out_stride =
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(k_chunk_count, k_chunk_length);
    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x21, %x[out]\n"
        "mov x20, %x[width]\n"
        "ptrue p1.b\n"
        "1:"  // Bias: Full loop
        "whilelt p0.h, XZR, x20\n"
        "dech x20\n"
        "cmp x20, #0x0\n"
        "ld1h { z16.h }, p0/Z, [%x[bias]]\n"
        "incb %x[bias]\n"
        "st1h { z16.h }, p1, [x21]\n"
        "add x21, x21, %x[out_stride]\n"
        "bgt 1b\n"
        "incb %x[out]\n"
        "mov x11, %x[k_chunk_count]\n"
        "2:"  // Chunk Loop
        "mov x10, %x[height]\n"
        "cmp x10, #0x8\n"
        "blt 6f\n"
        "3:"  // Main row loop: Head
        "mov x9, %x[in]\n"
        "mov x28, %x[out]\n"
        "add x27, x9, %x[in_stride]\n"
        "sub x10, x10, #0x8\n"
        "add x26, x27, %x[in_stride]\n"
        "mov x25, %x[width]\n"
        "add x24, x26, %x[in_stride]\n"
        "add x23, x24, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "add %x[in], x20, %x[in_stride]\n"
        "4:"  // Main row loop: Column loop
        "whilelt p0.h, XZR, x25\n"
        "decw x25, ALL, MUL #2\n"
        "ld1h { z20.h }, p0/Z, [x9]\n"
        "cmp x25, #0x0\n"
        "addvl x9, x9, #1\n"
        "ld1h { z17.h }, p0/Z, [x27]\n"
        "addvl x27, x27, #1\n"
        "ld1h { z19.h }, p0/Z, [x26]\n"
        "addvl x26, x26, #1\n"
        "ld1h { z16.h }, p0/Z, [x24]\n"
        "addvl x24, x24, #1\n"
        "ld1h { z18.h }, p0/Z, [x23]\n"
        "addvl x23, x23, #1\n"
        "zip1 z24.h, z20.h, z17.h\n"
        "zip2 z23.h, z20.h, z17.h\n"
        "ld1h { z17.h }, p0/Z, [x22]\n"
        "addvl x22, x22, #1\n"
        "ld1h { z22.h }, p0/Z, [x21]\n"
        "addvl x21, x21, #1\n"
        "zip1 z21.h, z19.h, z16.h\n"
        "zip2 z20.h, z19.h, z16.h\n"
        "ld1h { z16.h }, p0/Z, [x20]\n"
        "addvl x20, x20, #1\n"
        "zip1 z19.h, z18.h, z17.h\n"
        "zip2 z18.h, z18.h, z17.h\n"
        "st1h { z24.h }, p1, [x28]\n"
        "st1h { z23.h }, p1, [x28, #1, MUL VL]\n"
        "zip1 z17.h, z22.h, z16.h\n"
        "zip2 z16.h, z22.h, z16.h\n"
        "st1h { z21.h }, p1, [x28, #2, MUL VL]\n"
        "st1h { z20.h }, p1, [x28, #3, MUL VL]\n"
        "st1h { z19.h }, p1, [x28, #4, MUL VL]\n"
        "st1h { z18.h }, p1, [x28, #5, MUL VL]\n"
        "st1h { z17.h }, p1, [x28, #6, MUL VL]\n"
        "st1h { z16.h }, p1, [x28, #7, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 4b\n"
        "cmp x10, #0x8\n"
        "addvl %x[out], %x[out], #8\n"
        "bge 3b\n"
        "cbz x10, 10f\n"
        "6:"  // Main loop skip
        "7:"  // Tail row loop: Head
        "mov x9, %x[in]\n"
        "cntw x22, ALL, MUL #4\n"
        "add x27, x9, %x[in_stride]\n"
        "cmp x10, #0x1\n"
        "add %x[in], x27, %x[in_stride]\n"
        "mov x28, %x[out]\n"
        "csel %x[in], %x[in], x27, GT\n"
        "csel x27, x27, %x[pad_row], GT\n"
        "csel x21, x22, XZR, GT\n"
        "sub x10, x10, #0x2\n"
        "mov x20, %x[width]\n"
        "8:"  // Tail row loop: Column loop
        "whilelt p0.h, XZR, x20\n"
        "decw x20, ALL, MUL #2\n"
        "ld1h { z18.h }, p0/Z, [x9]\n"
        "cmp x20, #0x0\n"
        "add x9, x9, x22\n"
        "ld1h { z16.h }, p0/Z, [x27]\n"
        "add x27, x27, x21\n"
        "zip1 z17.h, z18.h, z16.h\n"
        "zip2 z16.h, z18.h, z16.h\n"
        "st1h { z17.h }, p1, [x28]\n"
        "st1h { z16.h }, p1, [x28, #1, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 8b\n"
        "cmp x10, #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 7b\n"
        "10:"  // Done
        "sub x11, x11, #0x1\n"
        "cbnz x11, 2b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [in] "+&r"(in), [out] "+&r"(out)
        : [height] "r"(height), [in_stride] "r"(in_stride), [k_chunk_count] "r"(k_chunk_count),
          [out_stride] "r"(out_stride), [pad_row] "r"(pad_row), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1",
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24",
          "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
