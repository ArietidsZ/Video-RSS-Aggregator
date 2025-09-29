use std::env;
use std::path::PathBuf;

fn main() {
    // Determine build configuration
    let cuda_available = cfg!(feature = "cuda");
    let rocm_available = cfg!(feature = "rocm");
    let metal_available = cfg!(feature = "metal");

    // Build C++ library with CMake
    let dst = cmake::Config::new(".")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("USE_CUDA", if cuda_available { "ON" } else { "OFF" })
        .define("USE_ROCM", if rocm_available { "ON" } else { "OFF" })
        .define("USE_METAL", if metal_available { "ON" } else { "OFF" })
        .build();

    // Link the built library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=whisper_turbo");

    // Link CUDA libraries if available
    if cuda_available {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
        println!("cargo:rustc-link-lib=cufft");

        // Find CUDA installation
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    }

    // Link ROCm libraries if available
    if rocm_available {
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=rocsparse");
        println!("cargo:rustc-link-lib=rocfft");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    }

    // Link Metal Performance Shaders for Apple Silicon
    if metal_available {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }

    // Link CTranslate2
    println!("cargo:rustc-link-lib=ctranslate2");

    // Link oneDNN (Intel MKL-DNN)
    println!("cargo:rustc-link-lib=dnnl");

    // Link OpenMP if available
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=gomp");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=omp");
    }

    // Generate bindings (optional, for development)
    #[cfg(feature = "bindgen")]
    {
        let bindings = bindgen::Builder::default()
            .header("include/whisper_turbo.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}