//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_test_common.hpp"

#include <iostream>
#include <ostream>

namespace kai::test {

std::ostream& operator<<(std::ostream& os, const MatMulShape& shape) {
    return os << "[m=" << shape.m << ", n=" << shape.n << ", k=" << shape.k << "]";
}

void PrintTo(const MatMulTestParams& param, std::ostream* os) {
    const auto& [method, shape, portion] = param;

    *os << "Method_" << method.name << "__";
    PrintTo(shape, os);
    *os << "__";
    PrintTo(portion, os);
}

void PrintTo(const MatMulShape& shape, std::ostream* os) {
    *os << "M_" << shape.m << "__N_" << shape.n << "__K_" << shape.k;
}

void PrintTo(const MatrixPortion& portion, std::ostream* os) {
    *os << "PortionStartRow_" << static_cast<int>(portion.start_row() * 1000)    //
        << "__PortionStartCol_" << static_cast<int>(portion.start_col() * 1000)  //
        << "__PortionHeight_" << static_cast<int>(portion.height() * 1000)       //
        << "__PortionWidth_" << static_cast<int>(portion.width() * 1000);
}
}  // namespace kai::test
