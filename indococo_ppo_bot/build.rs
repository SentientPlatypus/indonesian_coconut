fn main() {
    println!("cargo:rerun-if-changed=src/ripple_tf32.cpp");
    println!("cargo:rerun-if-env-changed=RIPPLE_CUDA_INCLUDE");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if std::env::var("CARGO_FEATURE_TORCH").is_err() {
        return;
    }
    // torch-sys sets links = "tch", so cargo exports DEP_TCH_* to dependents.
    let lib = std::env::var("DEP_TCH_LIBTORCH_LIB")
        .or_else(|_| std::env::var("DEP_TORCH_SYS_LIBTORCH_LIB"))
        .expect("torch-sys did not export DEP_TCH_LIBTORCH_LIB or DEP_TORCH_SYS_LIBTORCH_LIB");
    let lib_dir = std::path::PathBuf::from(&lib);
    let root = lib_dir.parent().expect("libtorch lib dir has a parent");
    let abi = std::env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_string());
    let cuda = lib_dir.join("libtorch_cuda.so").exists() || lib_dir.join("libc10_cuda.so").exists();
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/ripple_tf32.cpp")
        .include(root.join("include"))
        .include(root.join("include/torch/csrc/api/include"))
        .flag("-std=c++17")
        .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={abi}"))
        .flag("-DGLOG_USE_GLOG_EXPORT");
    if cuda {
        build.flag("-DRIPPLE_LIBTORCH_CUDA");
        if let Some(include) = cuda_include() {
            build.include(&include);
            println!("cargo:rerun-if-changed={}", include.display());
        }
    }
    build.compile("ripple_tf32");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if cuda {
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }
}

/// CUDA runtime headers for `ripple_tf32.cpp`. `RIPPLE_CUDA_INCLUDE` is the
/// include directory. `CUDA_HOME` and `CUDA_PATH` are toolkit roots.
fn cuda_include() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("RIPPLE_CUDA_INCLUDE") {
        return Some(std::path::PathBuf::from(dir));
    }
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(var) {
            return Some(std::path::PathBuf::from(root).join("include"));
        }
    }
    let default = std::path::PathBuf::from("/usr/local/cuda/include");
    if default.is_dir() {
        Some(default)
    } else {
        None
    }
}
