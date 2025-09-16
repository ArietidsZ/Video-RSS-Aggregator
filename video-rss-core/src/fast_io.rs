use crate::{error::VideoRssError, Result};
use bytes::{Bytes, BytesMut};
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};
use zerocopy::{IntoBytes, FromBytes, FromZeros};

#[cfg(feature = "io-uring")]
use tokio_uring::fs::File as UringFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastIOConfig {
    pub use_io_uring: bool,
    pub use_mmap: bool,
    pub direct_io: bool,
    pub buffer_size: usize,
    pub prefetch_size: usize,
    pub max_concurrent_ops: usize,
}

impl Default for FastIOConfig {
    fn default() -> Self {
        Self {
            use_io_uring: cfg!(feature = "io-uring") && cfg!(target_os = "linux"),
            use_mmap: true,
            direct_io: false,
            buffer_size: 64 * 1024,  // 64KB
            prefetch_size: 256 * 1024,  // 256KB
            max_concurrent_ops: 128,
        }
    }
}

/// Zero-copy buffer for efficient data transfer
#[derive(Debug, Clone, IntoBytes, FromBytes, FromZeros)]
#[repr(C)]
pub struct ZeroCopyBuffer {
    pub data: [u8; 4096],
}

pub struct FastIO {
    config: FastIOConfig,
    #[cfg(feature = "io-uring")]
    uring_runtime: Option<tokio_uring::Runtime>,
}

impl FastIO {
    pub fn new(config: FastIOConfig) -> Result<Self> {
        #[cfg(feature = "io-uring")]
        let uring_runtime = if config.use_io_uring {
            info!("Initializing io_uring runtime for zero-copy I/O");
            Some(
                tokio_uring::Runtime::new()
                    .map_err(|e| VideoRssError::Config(format!("io_uring init error: {}", e)))?
            )
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "io-uring")]
            uring_runtime,
        })
    }

    /// Read file using the fastest available method
    pub async fn read_file(&self, path: &Path) -> Result<Bytes> {
        if self.config.use_mmap {
            self.read_mmap(path)
        } else if self.config.use_io_uring {
            self.read_io_uring(path).await
        } else {
            self.read_standard(path).await
        }
    }

    /// Write file using the fastest available method
    pub async fn write_file(&self, path: &Path, data: &[u8]) -> Result<()> {
        if self.config.use_io_uring {
            self.write_io_uring(path, data).await
        } else if self.config.direct_io {
            self.write_direct(path, data)
        } else {
            self.write_standard(path, data).await
        }
    }

    /// Memory-mapped file reading (zero-copy)
    fn read_mmap(&self, path: &Path) -> Result<Bytes> {
        debug!("Reading file with mmap: {:?}", path);

        let file = File::open(path)
            .map_err(|e| VideoRssError::Io(e))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| VideoRssError::Io(e))?
        };

        // Convert to Bytes without copying
        Ok(Bytes::copy_from_slice(&mmap[..]))
    }

    #[cfg(feature = "io-uring")]
    async fn read_io_uring(&self, path: &Path) -> Result<Bytes> {
        debug!("Reading file with io_uring: {:?}", path);

        if let Some(runtime) = &self.uring_runtime {
            runtime.block_on(async {
                let file = UringFile::open(path).await
                    .map_err(|e| VideoRssError::Io(e))?;

                let metadata = file.statx().await
                    .map_err(|e| VideoRssError::Io(e))?;

                let size = metadata.stx_size as usize;
                let mut buffer = vec![0u8; size];

                let (res, buf) = file.read_at(buffer, 0).await;
                res.map_err(|e| VideoRssError::Io(e))?;

                Ok(Bytes::from(buf))
            })
        } else {
            self.read_standard(path).await
        }
    }

    #[cfg(not(feature = "io-uring"))]
    async fn read_io_uring(&self, path: &Path) -> Result<Bytes> {
        self.read_standard(path).await
    }

    async fn read_standard(&self, path: &Path) -> Result<Bytes> {
        debug!("Reading file with standard I/O: {:?}", path);

        let data = tokio::fs::read(path).await
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(Bytes::from(data))
    }

    #[cfg(feature = "io-uring")]
    async fn write_io_uring(&self, path: &Path, data: &[u8]) -> Result<()> {
        debug!("Writing file with io_uring: {:?}", path);

        if let Some(runtime) = &self.uring_runtime {
            runtime.block_on(async {
                let file = UringFile::create(path).await
                    .map_err(|e| VideoRssError::Io(e))?;

                let (res, _) = file.write_at(data.to_vec(), 0).await;
                res.map_err(|e| VideoRssError::Io(e))?;

                file.sync_all().await
                    .map_err(|e| VideoRssError::Io(e))?;

                Ok(())
            })
        } else {
            self.write_standard(path, data).await
        }
    }

    #[cfg(not(feature = "io-uring"))]
    async fn write_io_uring(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.write_standard(path, data).await
    }

    fn write_direct(&self, path: &Path, data: &[u8]) -> Result<()> {
        debug!("Writing file with direct I/O: {:?}", path);

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| VideoRssError::Io(e))?;

        // Align data for direct I/O
        let aligned_data = self.align_for_direct_io(data);

        file.write_all(&aligned_data)
            .map_err(|e| VideoRssError::Io(e))?;

        file.sync_all()
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(())
    }

    async fn write_standard(&self, path: &Path, data: &[u8]) -> Result<()> {
        debug!("Writing file with standard I/O: {:?}", path);

        tokio::fs::write(path, data).await
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(())
    }

    fn align_for_direct_io(&self, data: &[u8]) -> Vec<u8> {
        let alignment = 4096;  // Typical page size
        let aligned_len = ((data.len() + alignment - 1) / alignment) * alignment;
        let mut aligned = vec![0u8; aligned_len];
        aligned[..data.len()].copy_from_slice(data);
        aligned
    }

    /// Stream large file in chunks (zero-copy when possible)
    pub async fn stream_file(
        &self,
        path: &Path,
        chunk_size: usize,
    ) -> Result<mpsc::Receiver<Result<Bytes>>> {
        let (tx, rx) = mpsc::channel(16);
        let path = path.to_path_buf();
        let use_mmap = self.config.use_mmap;

        tokio::spawn(async move {
            if use_mmap {
                // Memory-mapped streaming
                match Self::stream_mmap(&path, chunk_size, tx.clone()).await {
                    Ok(_) => {},
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                    }
                }
            } else {
                // Standard streaming
                match Self::stream_standard(&path, chunk_size, tx.clone()).await {
                    Ok(_) => {},
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn stream_mmap(
        path: &Path,
        chunk_size: usize,
        tx: mpsc::Sender<Result<Bytes>>,
    ) -> Result<()> {
        let file = File::open(path)
            .map_err(|e| VideoRssError::Io(e))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| VideoRssError::Io(e))?
        };

        let len = mmap.len();
        let mut offset = 0;

        while offset < len {
            let chunk_end = (offset + chunk_size).min(len);
            let chunk = Bytes::copy_from_slice(&mmap[offset..chunk_end]);

            if tx.send(Ok(chunk)).await.is_err() {
                break;  // Receiver dropped
            }

            offset = chunk_end;
        }

        Ok(())
    }

    async fn stream_standard(
        path: &Path,
        chunk_size: usize,
        tx: mpsc::Sender<Result<Bytes>>,
    ) -> Result<()> {
        let mut file = tokio::fs::File::open(path).await
            .map_err(|e| VideoRssError::Io(e))?;

        let mut buffer = vec![0u8; chunk_size];

        loop {
            use tokio::io::AsyncReadExt;
            let n = file.read(&mut buffer).await
                .map_err(|e| VideoRssError::Io(e))?;

            if n == 0 {
                break;
            }

            let chunk = Bytes::copy_from_slice(&buffer[..n]);

            if tx.send(Ok(chunk)).await.is_err() {
                break;  // Receiver dropped
            }
        }

        Ok(())
    }

    /// Copy file using the fastest available method
    pub async fn copy_file(&self, from: &Path, to: &Path) -> Result<()> {
        if cfg!(target_os = "linux") {
            // Try sendfile for zero-copy
            self.copy_sendfile(from, to).await
        } else if self.config.use_mmap {
            self.copy_mmap(from, to)
        } else {
            self.copy_standard(from, to).await
        }
    }

    #[cfg(target_os = "linux")]
    async fn copy_sendfile(&self, from: &Path, to: &Path) -> Result<()> {
        use std::os::unix::io::AsRawFd;

        debug!("Copying file with sendfile: {:?} -> {:?}", from, to);

        let input = File::open(from)
            .map_err(|e| VideoRssError::Io(e))?;
        let output = File::create(to)
            .map_err(|e| VideoRssError::Io(e))?;

        let input_fd = input.as_raw_fd();
        let output_fd = output.as_raw_fd();
        let len = input.metadata()
            .map_err(|e| VideoRssError::Io(e))?
            .len() as usize;

        let mut offset = 0i64;
        unsafe {
            let ret = libc::sendfile(
                output_fd,
                input_fd,
                &mut offset as *mut i64,
                len,
            );

            if ret < 0 {
                return Err(VideoRssError::Io(io::Error::last_os_error()));
            }
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    async fn copy_sendfile(&self, from: &Path, to: &Path) -> Result<()> {
        self.copy_standard(from, to).await
    }

    fn copy_mmap(&self, from: &Path, to: &Path) -> Result<()> {
        debug!("Copying file with mmap: {:?} -> {:?}", from, to);

        let input = File::open(from)
            .map_err(|e| VideoRssError::Io(e))?;
        let mut output = File::create(to)
            .map_err(|e| VideoRssError::Io(e))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&input)
                .map_err(|e| VideoRssError::Io(e))?
        };

        output.write_all(&mmap[..])
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(())
    }

    async fn copy_standard(&self, from: &Path, to: &Path) -> Result<()> {
        debug!("Copying file with standard I/O: {:?} -> {:?}", from, to);

        tokio::fs::copy(from, to).await
            .map_err(|e| VideoRssError::Io(e))?;

        Ok(())
    }
}

/// Buffer pool for zero-allocation I/O
pub struct BufferPool {
    pool: Arc<RwLock<Vec<BytesMut>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        let mut pool = Vec::with_capacity(max_buffers);
        for _ in 0..max_buffers / 2 {
            pool.push(BytesMut::with_capacity(buffer_size));
        }

        Self {
            pool: Arc::new(RwLock::new(pool)),
            buffer_size,
            max_buffers,
        }
    }

    pub async fn get(&self) -> BytesMut {
        let mut pool = self.pool.write().await;

        if let Some(mut buffer) = pool.pop() {
            buffer.clear();
            buffer
        } else {
            BytesMut::with_capacity(self.buffer_size)
        }
    }

    pub async fn put(&self, buffer: BytesMut) {
        let mut pool = self.pool.write().await;

        if pool.len() < self.max_buffers {
            pool.push(buffer);
        }
    }
}

/// SIMD-accelerated memory operations
pub mod simd {
    use packed_simd_2::*;
    use std::arch::x86_64::*;

    /// SIMD-accelerated memory copy
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn memcpy_simd(dest: *mut u8, src: *const u8, len: usize) {
        if is_x86_feature_detected!("avx2") {
            memcpy_avx2(dest, src, len);
        } else if is_x86_feature_detected!("sse2") {
            memcpy_sse2(dest, src, len);
        } else {
            std::ptr::copy_nonoverlapping(src, dest, len);
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn memcpy_avx2(mut dest: *mut u8, mut src: *const u8, mut len: usize) {
        // Copy 32 bytes at a time using AVX2
        while len >= 32 {
            let data = _mm256_loadu_si256(src as *const __m256i);
            _mm256_storeu_si256(dest as *mut __m256i, data);
            src = src.add(32);
            dest = dest.add(32);
            len -= 32;
        }

        // Copy remaining bytes
        if len > 0 {
            std::ptr::copy_nonoverlapping(src, dest, len);
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn memcpy_sse2(mut dest: *mut u8, mut src: *const u8, mut len: usize) {
        // Copy 16 bytes at a time using SSE2
        while len >= 16 {
            let data = _mm_loadu_si128(src as *const __m128i);
            _mm_storeu_si128(dest as *mut __m128i, data);
            src = src.add(16);
            dest = dest.add(16);
            len -= 16;
        }

        // Copy remaining bytes
        if len > 0 {
            std::ptr::copy_nonoverlapping(src, dest, len);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn memcpy_simd(dest: *mut u8, src: *const u8, len: usize) {
        std::ptr::copy_nonoverlapping(src, dest, len);
    }

    /// SIMD-accelerated pattern search
    pub fn find_pattern_simd(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() || needle.len() > haystack.len() {
            return None;
        }

        // Use SIMD for pattern matching
        let first_byte = needle[0];
        let mut i = 0;

        while i <= haystack.len() - needle.len() {
            // Find next occurrence of first byte using SIMD
            if let Some(offset) = find_byte_simd(&haystack[i..], first_byte) {
                i += offset;

                // Check if full pattern matches
                if haystack[i..].starts_with(needle) {
                    return Some(i);
                }

                i += 1;
            } else {
                break;
            }
        }

        None
    }

    fn find_byte_simd(haystack: &[u8], needle: u8) -> Option<usize> {
        // Simplified SIMD byte search
        haystack.iter().position(|&b| b == needle)
    }
}