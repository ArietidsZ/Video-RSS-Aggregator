// FFI bridge for Go integration
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::{VideoProcessor, VideoAnalysisConfig, GpuBackendType};

/// C-compatible result structure
#[repr(C)]
pub struct FFIResult {
    pub success: bool,
    pub data: *mut c_char,
    pub error: *mut c_char,
}

/// Process video from URL - called from Go
#[no_mangle]
pub extern "C" fn process_video(
    video_path: *const c_char,
    gpu_backend: *const c_char,
    max_frames: c_int,
) -> FFIResult {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        // Parse input parameters
        let path = unsafe {
            if video_path.is_null() {
                return FFIResult {
                    success: false,
                    data: ptr::null_mut(),
                    error: CString::new("video_path is null").unwrap().into_raw(),
                };
            }
            CStr::from_ptr(video_path).to_string_lossy().to_string()
        };

        let backend = unsafe {
            if gpu_backend.is_null() {
                "auto"
            } else {
                CStr::from_ptr(gpu_backend).to_str().unwrap_or("auto")
            }
        };

        // Configure processor
        let config = VideoAnalysisConfig {
            gpu_backend: match backend {
                "cuda" => GpuBackendType::Cuda,
                "rocm" => GpuBackendType::Rocm,
                "openvino" => GpuBackendType::OpenVino,
                "metal" => GpuBackendType::Metal,
                "cpu" => GpuBackendType::Cpu,
                _ => GpuBackendType::Auto,
            },
            max_frames: max_frames as usize,
            ..Default::default()
        };

        // Process video
        match VideoProcessor::new(config).await {
            Ok(processor) => {
                match processor.process_video(Path::new(&path)).await {
                    Ok(result) => {
                        // Serialize result to JSON
                        match serde_json::to_string(&result) {
                            Ok(json) => FFIResult {
                                success: true,
                                data: CString::new(json).unwrap().into_raw(),
                                error: ptr::null_mut(),
                            },
                            Err(e) => FFIResult {
                                success: false,
                                data: ptr::null_mut(),
                                error: CString::new(format!("Serialization error: {}", e))
                                    .unwrap()
                                    .into_raw(),
                            },
                        }
                    }
                    Err(e) => FFIResult {
                        success: false,
                        data: ptr::null_mut(),
                        error: CString::new(format!("Processing error: {}", e))
                            .unwrap()
                            .into_raw(),
                    },
                }
            }
            Err(e) => FFIResult {
                success: false,
                data: ptr::null_mut(),
                error: CString::new(format!("Initialization error: {}", e))
                    .unwrap()
                    .into_raw(),
            },
        }
    })
}

/// Transcribe audio - called from Go
#[no_mangle]
pub extern "C" fn transcribe_audio(
    audio_data: *const f32,
    audio_len: usize,
    language: *const c_char,
) -> FFIResult {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        // Convert audio data from C array
        let audio = unsafe {
            if audio_data.is_null() {
                return FFIResult {
                    success: false,
                    data: ptr::null_mut(),
                    error: CString::new("audio_data is null").unwrap().into_raw(),
                };
            }
            slice::from_raw_parts(audio_data, audio_len).to_vec()
        };

        let lang = unsafe {
            if language.is_null() {
                "auto"
            } else {
                CStr::from_ptr(language).to_str().unwrap_or("auto")
            }
        };

        // Create processor with default config
        let config = VideoAnalysisConfig::default();

        match VideoProcessor::new(config).await {
            Ok(processor) => {
                match processor.transcriber.transcribe(&audio).await {
                    Ok(transcript) => {
                        match serde_json::to_string(&transcript) {
                            Ok(json) => FFIResult {
                                success: true,
                                data: CString::new(json).unwrap().into_raw(),
                                error: ptr::null_mut(),
                            },
                            Err(e) => FFIResult {
                                success: false,
                                data: ptr::null_mut(),
                                error: CString::new(format!("Serialization error: {}", e))
                                    .unwrap()
                                    .into_raw(),
                            },
                        }
                    }
                    Err(e) => FFIResult {
                        success: false,
                        data: ptr::null_mut(),
                        error: CString::new(format!("Transcription error: {}", e))
                            .unwrap()
                            .into_raw(),
                    },
                }
            }
            Err(e) => FFIResult {
                success: false,
                data: ptr::null_mut(),
                error: CString::new(format!("Initialization error: {}", e))
                    .unwrap()
                    .into_raw(),
            },
        }
    })
}

/// Get available GPU backends - called from Go
#[no_mangle]
pub extern "C" fn get_gpu_backends() -> *mut c_char {
    let backends = vec![
        #[cfg(feature = "cuda")]
        "cuda",
        #[cfg(feature = "rocm")]
        "rocm",
        #[cfg(feature = "openvino")]
        "openvino",
        #[cfg(target_os = "macos")]
        "metal",
        "cpu",
    ];

    let json = serde_json::to_string(&backends).unwrap();
    CString::new(json).unwrap().into_raw()
}

/// Free string allocated by Rust - must be called from Go
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Free FFIResult - must be called from Go
#[no_mangle]
pub extern "C" fn free_result(result: FFIResult) {
    if !result.data.is_null() {
        free_string(result.data);
    }
    if !result.error.is_null() {
        free_string(result.error);
    }
}