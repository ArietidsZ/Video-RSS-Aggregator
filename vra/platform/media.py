"""ffmpeg frame extraction: ONE decode pass, scene+uniform blend, dHash dedup.

OWNER: agent-A. Implements contracts.FrameExtractor (FfmpegFrameExtractor).
Requirements:
- Single ffmpeg invocation: split filtergraph [v_scene] (select=gt(scene,T) + fps
  prefilter) and [v_uniform] (fps=duration/max_frames grid, thumbnail=n filter for
  representative frame) -> two image2 outputs, -hwaccel auto, scale to
  frame_max_dim, mjpeg q=3. No second full decode.
- duration via ffprobe. mode auto: scene candidates >= min(3, max_frames) else blend.
- dHash 64-bit (9x8 grayscale) computed in-thread via ffmpeg scale to 9x8 gray
  rawvideo piped out (no Pillow), hex string. Near-dup hamming <= 6 dropped.
- Rank kept frames: scene_score desc within timeline-spread guarantee (pick evenly
  over ts). Returns FrameSet with ts_s timestamps (use -frame_pts or select metadata).
"""
