// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_aquant_pipeline_util.hpp"

using F16       = ck_tile::half_t;
using F32       = float;
using F8        = ck_tile::fp8_t;
using BF8        = ck_tile::bf8_t;
using I4        = ck_tile::pk_int4_t;
using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using Aquant    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Aquant>;

using KernelTypesAQuantCompV3 = ::testing::Types<
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, F8, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, BF8, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F32, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, F8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<128>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<64>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<32>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<16>, ck_tile::bool_constant<false>, ck_tile::bool_constant<false>>,

    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<128>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<64>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<32>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>,
    std::tuple<Row, Col, Row, I4, BF8, F32, F32, Intrawave, Aquant, Row, F8, ck_tile::number<16>, ck_tile::bool_constant<true>, ck_tile::bool_constant<false>>>;

// clang-format on
