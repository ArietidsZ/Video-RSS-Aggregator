use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use libloading::Library;

use crate::accel::AccelBackendKind;

type InitFn = unsafe extern "C" fn(config_json: *const std::os::raw::c_char) -> i32;
type TranscribeFn = unsafe extern "C" fn(
    audio_path: *const std::os::raw::c_char,
    output_json: *mut *const std::os::raw::c_char,
) -> i32;
type SummarizeFn = unsafe extern "C" fn(
    text: *const std::os::raw::c_char,
    output_json: *mut *const std::os::raw::c_char,
) -> i32;
type FreeFn = unsafe extern "C" fn(ptr: *const std::os::raw::c_char);

pub struct BackendLibrary {
    _lib: Library,
    init: InitFn,
    transcribe: TranscribeFn,
    summarize: SummarizeFn,
    free_string: FreeFn,
}

impl BackendLibrary {
    pub fn load(
        kind: AccelBackendKind,
        lib_dir: Option<PathBuf>,
        lib_name: Option<String>,
        config_json: &serde_json::Value,
    ) -> Result<Self> {
        let lib_path = resolve_library_path(kind, lib_dir, lib_name)?;
        let lib = unsafe { Library::new(&lib_path) }.map_err(|err| {
            anyhow!(
                "Failed to load backend library {}: {}",
                lib_path.display(),
                err
            )
        })?;

        let init: InitFn = unsafe { *lib.get(b"vra_backend_init\0")? };
        let transcribe: TranscribeFn = unsafe { *lib.get(b"vra_backend_transcribe\0")? };
        let summarize: SummarizeFn = unsafe { *lib.get(b"vra_backend_summarize\0")? };
        let free_string: FreeFn = unsafe { *lib.get(b"vra_backend_free_string\0")? };

        let config_text = serde_json::to_string(config_json)?;
        let config_cstr = CString::new(config_text)?;
        let init_status = unsafe { init(config_cstr.as_ptr()) };
        if init_status != 0 {
            return Err(anyhow!("Backend init failed with status {}", init_status));
        }

        Ok(Self {
            _lib: lib,
            init,
            transcribe,
            summarize,
            free_string,
        })
    }

    pub fn transcribe(&self, audio_path: &str) -> Result<String> {
        self.call_with_output(self.transcribe, audio_path)
    }

    pub fn summarize(&self, text: &str) -> Result<String> {
        self.call_with_output(self.summarize, text)
    }

    fn call_with_output(
        &self,
        func: unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *mut *const std::os::raw::c_char,
        ) -> i32,
        input: &str,
    ) -> Result<String> {
        let input_c = CString::new(input)?;
        let mut output_ptr: *const std::os::raw::c_char = std::ptr::null();

        let status = unsafe { func(input_c.as_ptr(), &mut output_ptr as *mut _) };
        if status != 0 {
            return Err(anyhow!("Backend call failed with status {}", status));
        }
        if output_ptr.is_null() {
            return Err(anyhow!("Backend returned null output"));
        }

        let output = unsafe { CStr::from_ptr(output_ptr) }
            .to_string_lossy()
            .to_string();
        unsafe { (self.free_string)(output_ptr) };

        Ok(output)
    }
}

fn resolve_library_path(
    kind: AccelBackendKind,
    lib_dir: Option<PathBuf>,
    lib_name: Option<String>,
) -> Result<PathBuf> {
    let name = if let Some(name) = lib_name {
        name
    } else {
        default_library_name(kind).to_string()
    };

    let path = if let Some(dir) = lib_dir {
        dir.join(&name)
    } else {
        Path::new(&name).to_path_buf()
    };

    Ok(path)
}

fn default_library_name(kind: AccelBackendKind) -> &'static str {
    if cfg!(target_os = "windows") {
        match kind {
            AccelBackendKind::Cuda => "vra_cuda_backend.dll",
            AccelBackendKind::Rocm => "vra_rocm_backend.dll",
            AccelBackendKind::OneApi => "vra_oneapi_backend.dll",
            AccelBackendKind::Mps => "vra_mps_backend.dll",
            AccelBackendKind::CoreMl => "vra_coreml_backend.dll",
            AccelBackendKind::Cpu => "vra_cpu_backend.dll",
        }
    } else if cfg!(target_os = "macos") {
        match kind {
            AccelBackendKind::Cuda => "libvra_cuda_backend.dylib",
            AccelBackendKind::Rocm => "libvra_rocm_backend.dylib",
            AccelBackendKind::OneApi => "libvra_oneapi_backend.dylib",
            AccelBackendKind::Mps => "libvra_mps_backend.dylib",
            AccelBackendKind::CoreMl => "libvra_coreml_backend.dylib",
            AccelBackendKind::Cpu => "libvra_cpu_backend.dylib",
        }
    } else {
        match kind {
            AccelBackendKind::Cuda => "libvra_cuda_backend.so",
            AccelBackendKind::Rocm => "libvra_rocm_backend.so",
            AccelBackendKind::OneApi => "libvra_oneapi_backend.so",
            AccelBackendKind::Mps => "libvra_mps_backend.so",
            AccelBackendKind::CoreMl => "libvra_coreml_backend.so",
            AccelBackendKind::Cpu => "libvra_cpu_backend.so",
        }
    }
}
