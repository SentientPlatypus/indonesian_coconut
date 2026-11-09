// Host-side helpers for the Ripple LibTorch forward.
//
// TF32 is a process-wide cuBLAS flag. Ripple turns it off only while its own
// kernels are being launched, then restores the previous value so the learner
// update keeps whatever precision it had. Streams are per calling thread so
// two collector pools can launch at once without sharing the default stream.

#include <ATen/Context.h>

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

size_t ripple_cuda_allocated(int device) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (!at::globalContext().hasCUDA()) {
        return 0;
    }
    return c10::cuda::CUDACachingAllocator::currentMemoryAllocated(device);
#else
    (void)device;
    return 0;
#endif
}

size_t ripple_cuda_reserved(int device) {
#if defined(RIPPLE_LIBTORCH_CUDA)
    if (!at::globalContext().hasCUDA()) {
        return 0;
    }
    return c10::cuda::CUDACachingAllocator::currentMemoryReserved(device);
#else
    (void)device;
    return 0;
#endif
}

}
