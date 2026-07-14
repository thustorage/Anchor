#include "test_gemm_aquant_pipeline_kernel_types.hpp"
#include "test_gemm_aquant_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmAQuantPipelineCompV3 : public TestCkTileGemmPipeline<T>
{
};

#define TEST_SUITE_NAME TestCkTileGemmAQuantPipelineCompV3

TYPED_TEST_SUITE(TestCkTileGemmAQuantPipelineCompV3, KernelTypesAQuantCompV3);

#include "test_gemm_aquant_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME
