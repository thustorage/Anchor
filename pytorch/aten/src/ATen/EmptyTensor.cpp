#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/safe_numerics.h>

#include <limits>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>

#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>

namespace at::detail {

  namespace {
  
  static std::string g_ipc_socket_path = "/tmp/uipc_socket_v3";
  static std::atomic<uint64_t> g_ipc_alloc_id_counter{1};
  
  static constexpr uint8_t IPC_MAGIC = 0xAB;
  static constexpr uint8_t IPC_CMD_GENERIC = 0x01;
  static constexpr uint8_t IPC_CMD_STRIDED = 0x02;
  static constexpr uint8_t IPC_CMD_FREE    = 0x03;
  static constexpr uint8_t IPC_STATUS_OK   = 0x00;

  static std::string get_ipc_socket_path() {
    const char* env_path = std::getenv("PYTORCH_IPC_SOCKET_PATH");
    if (env_path && env_path[0] != '\0') {
      return std::string(env_path);
    }
    return g_ipc_socket_path;
  }
  
  
  static int ipc_socket_connect(const std::string& path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    TORCH_CHECK(fd >= 0, "IPC: socket() failed: ", strerror(errno));
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    int ret = ::connect(fd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0) {
      ::close(fd);
      TORCH_CHECK(false, "IPC: connect() to ", path, " failed: ", strerror(errno));
    }
    return fd;
  }
  
  static void ipc_socket_send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
      ssize_t n = ::send(fd, p + sent, len - sent, 0);
      TORCH_CHECK(n > 0, "IPC: send() failed: ", strerror(errno));
      sent += static_cast<size_t>(n);
    }
  }
  
  static void ipc_socket_recv_all(int fd, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    size_t received = 0;
    while (received < len) {
      ssize_t n = ::recv(fd, p + received, len - received, 0);
      TORCH_CHECK(n > 0, "IPC: recv() failed: ", strerror(errno));
      received += static_cast<size_t>(n);
    }
  }
  
  static void ipc_socket_send_frame(int fd, const std::vector<uint8_t>& buf) {
    uint32_t net_len = htonl(static_cast<uint32_t>(buf.size()));
    ipc_socket_send_all(fd, &net_len, 4);
    ipc_socket_send_all(fd, buf.data(), buf.size());
  }
  
  static std::vector<uint8_t> ipc_socket_recv_frame(int fd) {
    uint32_t net_len = 0;
    ipc_socket_recv_all(fd, &net_len, 4);
    uint32_t body_len = ntohl(net_len);
    std::vector<uint8_t> buf(body_len);
    ipc_socket_recv_all(fd, buf.data(), body_len);
    return buf;
  }
  
  
  static void encode_u8(std::vector<uint8_t>& buf, uint8_t v) {
    buf.push_back(v);
  }
  
  static void encode_u16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
  }
  
  static void encode_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8)  & 0xFF));
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
  }
  
  static void encode_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 7; i >= 0; --i) {
      buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
  }
  
  static void encode_i64(std::vector<uint8_t>& buf, int64_t v) {
    encode_u64(buf, static_cast<uint64_t>(v));
  }
  
  static uint8_t decode_u8(const uint8_t*& p) {
    return *p++;
  }
  
  static uint64_t decode_u64(const uint8_t*& p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v = (v << 8) | (*p++);
    }
    return v;
  }
  
  
  struct IpcDeleterContext {
    void* mapped_ptr;
    uint64_t alloc_id;
    std::string socket_path;
    ~IpcDeleterContext() {
      if (mapped_ptr) {
        cudaIpcCloseMemHandle(mapped_ptr);
      }
      try {
        const char* env_val = std::getenv("PYTORCH_NO_FREE");
        if (!env_val || std::strcmp(env_val, "0") == 0){
          int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
          if (fd < 0) return;
          struct sockaddr_un addr;
          std::memset(&addr, 0, sizeof(addr));
          addr.sun_family = AF_UNIX;
          std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
          if (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(fd);
            return;
          }
          std::vector<uint8_t> buf;
          encode_u8(buf, IPC_MAGIC);
          encode_u8(buf, IPC_CMD_FREE);
          encode_u64(buf, alloc_id);
          uint32_t net_len = htonl(static_cast<uint32_t>(buf.size()));
  
          ::send(fd, &net_len, 4, 0);
          ::send(fd, buf.data(), buf.size(), 0);
          ::close(fd);
        }
      } catch (...) {
      }
    }
  };
  
  static void ipc_deleter_fn(void* ctx_ptr) {
    delete static_cast<IpcDeleterContext*>(ctx_ptr);
  }
  
  
  static TensorBase ipc_alloc_tensor_via_daemon(
      IntArrayRef sizes,
      IntArrayRef strides,
      c10::DispatchKeySet ks,
      ScalarType scalar_type,
      size_t size_bytes,
      std::optional<c10::MemoryFormat> memory_format_opt) {
  
    bool is_strided = !strides.empty();
    uint8_t cmd = is_strided ? IPC_CMD_STRIDED : IPC_CMD_GENERIC;
    uint64_t alloc_id = g_ipc_alloc_id_counter.fetch_add(1);
  
    std::vector<uint8_t> req;
    encode_u8(req, IPC_MAGIC);
    encode_u8(req, cmd);
    encode_u64(req, alloc_id);
    encode_u32(req, static_cast<uint32_t>(sizes.size()));
    for (auto s : sizes) {
      encode_i64(req, s);
    }
    if (is_strided) {
      for (auto s : strides) {
        encode_i64(req, s);
      }
    }
    encode_u8(req, static_cast<uint8_t>(scalar_type));
    int dev_idx = 0;
    cudaGetDevice(&dev_idx);
    encode_u16(req, static_cast<uint16_t>(dev_idx));
    encode_u64(req, static_cast<uint64_t>(size_bytes));
  
    const std::string socket_path = get_ipc_socket_path();
    int fd = ipc_socket_connect(socket_path);
    ipc_socket_send_frame(fd, req);
  
    auto resp = ipc_socket_recv_frame(fd);
    ::close(fd);
  
    const uint8_t* p = resp.data();
    uint8_t resp_magic = decode_u8(p);
    uint8_t resp_status = decode_u8(p);
    TORCH_CHECK(resp_magic == IPC_MAGIC, "IPC: invalid response magic");
    TORCH_CHECK(resp_status == IPC_STATUS_OK, "IPC: daemon returned error status");
  
    cudaIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, p, sizeof(cudaIpcMemHandle_t));
    p += sizeof(cudaIpcMemHandle_t);
  
    uint64_t resp_alloc_id = decode_u64(p);
    uint64_t resp_size_bytes = decode_u64(p);
    (void)resp_alloc_id;
    void* mapped_ptr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&mapped_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(err == cudaSuccess, "IPC: cudaIpcOpenMemHandle failed: ", cudaGetErrorString(err));
  
    auto* ctx = new IpcDeleterContext{mapped_ptr, alloc_id, socket_path};

    c10::Device device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(dev_idx));
    auto data_ptr = c10::DataPtr(mapped_ptr, ctx, &ipc_deleter_fn, device);
  
    caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        static_cast<size_t>(resp_size_bytes),
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizeable=*/false);
  
    auto tensor = detail::make_tensor_base<TensorImpl>(
        std::move(storage_impl), ks, dtype);
  
    if (is_strided) {
      tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    } else {
      if (ks.has(c10::DispatchKey::Meta) || sizes.size() != 1 || sizes[0] != 0) {
        tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(sizes);
      }
      if (memory_format_opt.has_value()) {
        if (*memory_format_opt != MemoryFormat::Contiguous) {
          tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
        }
      }
    }
  
    return tensor;
  }
  
  }

bool isIPCForwardEnabled() {
  const char* env_val = std::getenv("PYTORCH_IPC_ALLOC");
  if (!env_val) {
    return false;
  }
  return (std::strcmp(env_val, "1") == 0 || std::strcmp(env_val, "true") == 0);
}



namespace {
c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    // NB: This is not quite right, if you somehow had both CUDA and PrivateUse1 initialized
    // in the same PyTorch build, you would ONLY ever get the CUDA pinned memory allocator.
    // To properly support this, see https://github.com/pytorch/pytorch/issues/14560

    std::optional<c10::DeviceType> opt_device_type = std::nullopt;
    // As mentioned in Note [Accelerator Context], the accelerators in PyTorch should be mutually exclusive,
    // and PrivateUse1 has the highest priority, followed by CUDA;
    // However, since exclusivity between accelerators cannot be guaranteed at present,
    // in order to ensure backward compatibility (previously the default was CUDA), CUDA are prioritized.
    if (at::globalContext().hasCUDA()) {
      opt_device_type = c10::DeviceType::CUDA;
    } else {
      opt_device_type = at::getAccelerator(false);
    }
    if (opt_device_type.has_value()) {
      return at::globalContext().getPinnedMemoryAllocator(opt_device_type);
    } else {
      TORCH_CHECK(
          false,
          "pin_memory=True requires a CUDA or other accelerator backend; "
          "no pinned memory allocator is available on this system.")
    }
  }

  return c10::GetCPUAllocator();
}

#ifndef C10_MOBILE
constexpr uint64_t storage_max() {
  // int64_t and size_t are used somewhat inconsistently throughout ATen.
  // To be safe, storage size calculations must fit in both types.
  constexpr auto int64_max = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max());
  constexpr auto size_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max());
  return std::min(int64_max, size_max);
}
#endif

inline void raise_warning_for_complex_half(ScalarType dtype) {
  if (dtype == kComplexHalf) {
    TORCH_WARN_ONCE(
        "ComplexHalf support is experimental and many operators don't support it yet.");
  }
}

}  // namespace (anonymous)

size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  uint64_t size = 1;
  bool overflowed = c10::safe_multiplies_u64(sizes, &size);
  overflowed |= c10::add_overflows(size, storage_offset, &size);
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=", sizes);
  return static_cast<size_t>(size);
#else
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
#endif
}

size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = storage_offset + 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    uint64_t strided_size = 0;
    overflowed |= c10::mul_overflows(strides[i], sizes[i] - 1, &strided_size);
    overflowed |= c10::add_overflows(size, strided_size, &size);
  }
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=",
              sizes, " and strides=", strides);
  return static_cast<size_t>(size);
#else
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
#endif
}

SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
}

// not including mobile-only macros in this function,
// since mobile shouldn't be using symints.
SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  SymInt size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (TORCH_GUARD_OR_FALSE(sizes[i].sym_eq(0))) {
      return 0;
    }

    // NOTE: while this can technically return negative sizes for
    // 0-element tensors, there's a check in TensorShape:set_storage_meta__symint
    // that skips setting nbytes with unbacked expressions.
    // Would probably be safer to wrap this with a max(*, 0),
    // once our min/max symbolic reasoning improves.
    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
}

template <typename T>
static TensorBase _empty_generic(
    ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);

  if constexpr (std::is_same_v<T, int64_t>) {
    if (ks.has(c10::DispatchKey::CUDA) && isIPCForwardEnabled()) {
      caffe2::TypeMeta dtype_ipc = scalarTypeToTypeMeta(scalar_type);
      auto size_bytes_ipc = computeStorageNbytesContiguous(size, dtype_ipc.itemsize());
      if (size_bytes_ipc > 0) {
        return ipc_alloc_tensor_via_daemon(
            size, /*strides=*/IntArrayRef{}, ks, scalar_type,
            size_bytes_ipc, memory_format_opt);
      }
    }
  }
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (ks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

TensorBase empty_generic_symint(
    SymIntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

template <typename T>
static TensorBase _empty_strided_generic(
    T size,
    T stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);
  if constexpr (std::is_same_v<T, IntArrayRef>) {
    if (ks.has(c10::DispatchKey::CUDA) && isIPCForwardEnabled()) {
      caffe2::TypeMeta dtype_ipc = scalarTypeToTypeMeta(scalar_type);
      auto size_bytes_ipc = computeStorageNbytes(size, stride, dtype_ipc.itemsize());
      if (size_bytes_ipc > 0) {
        return ipc_alloc_tensor_via_daemon(
            size, stride, ks, scalar_type,
            size_bytes_ipc, /*memory_format_opt=*/std::nullopt);
      }
    }
  }
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<IntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<SymIntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_cpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_cpu(IntArrayRef size, IntArrayRef stride,
                             ScalarType dtype, bool pin_memory) {
  auto allocator = at::detail::GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cpu_ks, dtype);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_cpu(size, stride, dtype, pin_memory);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_cpu(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

// The meta allocator ignores whatever allocation is requested and always
// gives you nullptr
struct MetaAllocator final : public at::Allocator {
  MetaAllocator() = default;
  ~MetaAllocator() override = default;
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  DataPtr allocate(const size_t nbytes [[maybe_unused]]) override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {}
};

static MetaAllocator g_meta_alloc;

REGISTER_ALLOCATOR(kMeta, &g_meta_alloc)

TensorBase empty_meta(IntArrayRef size, ScalarType dtype,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_generic(
      size, allocator, meta_dks, dtype, memory_format_opt);
}

TensorBase empty_meta(
  IntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "non-strided meta tensors not supported yet"
  );

  auto dtype = dtype_or_default(dtype_opt);
  return empty_meta(size, dtype, memory_format_opt);
}

TensorBase empty_symint_meta(
  SymIntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::Meta);
  auto scalar_type = dtype_or_default(dtype_opt);
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

TensorBase empty_meta(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_meta(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_meta(IntArrayRef size, IntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_meta(size, stride, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

TensorBase empty_strided_symint_meta(SymIntArrayRef size, SymIntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_symint_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_symint_meta(size, stride, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_symint_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt());
}

} // namespace at::detail
