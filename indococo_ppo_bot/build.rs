fn main() {
    println!("cargo:rerun-if-changed=src/ripple_tf32.cpp");
    if std::env::var("CARGO_FEATURE_TORCH").is_err() {
        return;
    }
    let lib = std::env::var("DEP_TORCH_SYS_LIBTORCH_LIB")
        .expect("torch-sys did not export DEP_TORCH_SYS_LIBTORCH_LIB");
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
    }
    build.compile("ripple_tf32");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
}
