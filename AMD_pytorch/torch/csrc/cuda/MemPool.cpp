#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::hip::MemPool>(torch_C_m, "_MemPool")
      .def(
          py::init([](c10::hip::HIPCachingAllocator::HIPAllocator* allocator,
                      bool is_user_created,
                      bool use_on_oom) {
            torch::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::c10::hip::MemPool>(
                allocator, is_user_created, use_on_oom);
          }))
      .def_property_readonly("id", &::c10::hip::MemPool::id)
      .def_property_readonly("allocator", &::c10::hip::MemPool::allocator)
      .def("use_count", &::c10::hip::MemPool::use_count);
}
