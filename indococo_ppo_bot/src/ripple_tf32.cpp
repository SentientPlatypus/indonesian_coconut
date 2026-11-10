// Host-side helpers for the Ripple LibTorch forward.
//
// TF32 is a process-wide cuBLAS flag. Ripple turns it off only while its own
// kernels are being launched, then restores the previous value so the learner
// update keeps whatever precision it had. Streams are per calling thread so
// two collector pools can launch at once without sharing the default stream.

#include <ATen/Context.h>
#include <ATen/Tensor.h>

#if defined(RIPPLE_LIBTORCH_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif

extern "C" {

int ripple_allow_tf32_cublas() {
    return at::globalContext().allowTF32CuBLAS() ? 1 : 0;
}

void ripple_set_allow_tf32_cublas(int enabled) {
    at::globalContext().setAllowTF32CuBLAS(enabled != 0);
}

int ripple_allow_tf32_cudnn() {
    return at::globalContext().allowTF32CuDNN() ? 1 : 0;
}

void ripple_set_allow_tf32_cudnn(int enabled) {
    at::globalContext().setAllowTF32CuDNN(enabled != 0);
}

void* ripple_stream_create(int device) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (!at::globalContext().hasCUDA()) {
        return nullptr;
    }
    try {
        auto stream = c10::cuda::getStreamFromPool(false, device);
        return new c10::cuda::CUDAStream(stream);
    } catch (...) {
        return nullptr;
    }
#else
    (void)device;
    return nullptr;
#endif
}

void ripple_stream_set(void* stream) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (stream != nullptr) {
        c10::cuda::setCurrentCUDAStream(*static_cast<c10::cuda::CUDAStream*>(stream));
    }
#else
    (void)stream;
#endif
}

// Put this thread back on the device default stream. League forwards switch
// to a side stream and call this before the collector's own stream is
// reapplied, so a thread that has no collector stream stays on the default.
void ripple_stream_restore_default(int device) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (!at::globalContext().hasCUDA()) {
        return;
    }
    auto stream = c10::cuda::getDefaultCUDAStream(static_cast<c10::DeviceIndex>(device));
    c10::cuda::setCurrentCUDAStream(stream);
#else
    (void)device;
#endif
}

// `dst` and `src` are tch `torch::Tensor*` values. The destination is pinned
// host memory. A blocking copy (non_blocking == 0) waits for the current
// stream inside libtorch. torch 2.9 implements c10::cuda::stream_synchronize
// and at::cuda::CUDAEvent::synchronize inline, and both call cudart directly,
// so this file must not call them.
void ripple_copy(void* dst, void* src, int non_blocking) {
    auto* destination = static_cast<at::Tensor*>(dst);
    auto* source = static_cast<at::Tensor*>(src);
    destination->copy_(*source, non_blocking != 0);
}

size_t ripple_cuda_stat(int device, bool reserved) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (!at::globalContext().hasCUDA()) {
        return 0;
    }
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
        static_cast<c10::DeviceIndex>(device));
    auto kind = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    if (reserved) {
        return stats.reserved_bytes[kind].current;
    }
    return stats.allocated_bytes[kind].current;
#else
    (void)device;
    (void)reserved;
    return 0;
#endif
}

size_t ripple_cuda_allocated(int device) {
    return ripple_cuda_stat(device, false);
}

size_t ripple_cuda_reserved(int device) {
    return ripple_cuda_stat(device, true);
}

}
