# Audio Watermark Sniffer

A professional, safe, and educational tool to visualize and detect hidden or ultrasonic tones in audio files that might be used as watermarks or advertising beacons.

This app provides:
- Tkinter GUI to load WAV/MP3 files
- Frequency spectrum visualization via matplotlib
- FFT/STFT-based analysis focused on ultrasonic range (>18 kHz)
- Adaptive detection with sensitivity control
- Report saving (TXT) and CSV export of detections

> Educational purpose only. Do not use to bypass or remove any copyright protection. The tool does not modify files.

## Features
- Load WAV natively; MP3 via `pydub` (requires ffmpeg)
- Toggle to view only ultrasonic band (>18 kHz)
- Sensitivity slider (dB above baseline) for detection tuning
- Detections summarized with time ranges and peak frequency/level
- Save TXT report and export detections to CSV

## Requirements
- Python 3.9+ (Windows/macOS/Linux)
- Recommended to use a virtual environment

Install dependencies:
```bash
pip install -r requirements.txt
```

For MP3 support, install ffmpeg and ensure it's on your PATH:
- Windows: download from `https://www.gyan.dev/ffmpeg/builds/` and add `ffmpeg\bin` to PATH
- macOS: `brew install ffmpeg`
- Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`

## Run
```bash
python main.py
```

## Using the App
1. Click "Load Audio" and choose a `.wav` or `.mp3` file.
2. Optionally toggle "Ultrasonic only (>18 kHz)" and adjust the sensitivity slider.
3. Click "Analyze". The right panel will list detected regions if any.
4. Use "Save Report" to save a TXT report in `analysis_reports/`.
5. Use "Export CSV" to save detections as CSV in `analysis_reports/`.

## How Detection Works (Brief)
- The app computes an STFT of the signal using a Hann window.
- It focuses on frequencies above 18 kHz (ultrasonic band by default).
- For each time frame, it computes the median level (dBFS relative to frame peak) within the ultrasonic band.
- A dynamic baseline is the median across time; frames exceeding baseline + sensitivity (dB) are flagged.
- Consecutive flagged frames are merged into detected regions; the peak frequency and level are reported.

This approach helps reveal persistent tones or bursts in the ultrasonic range that could indicate beacons or watermarks.

## Interpreting Results
- Peak frequency shows where the strongest ultrasonic component occurred in a region.
- dB values are relative to per-frame peak (dBFS). Higher positive numbers indicate stronger ultrasonic content versus the frame baseline.
- False positives can occur, especially on noisy or highly compressed content—adjust sensitivity to tune.

## Project Structure
```
audio-watermark-sniffer/
├─ main.py                  # main GUI + logic
├─ README.md                # instructions, explanation
├─ requirements.txt         # dependencies
├─ sample_audio/            # optional audio files for testing
└─ analysis_reports/        # auto-generated reports
```

## Screenshots / Demo
- Add screenshots or a GIF of the UI analyzing a sample file here.

## Safety & Ethics
- This tool is for analysis and education only.
- It does not remove or alter watermarks.
- Respect local laws and content owners' rights.

## Troubleshooting
- If imports fail (numpy/scipy/matplotlib), ensure the virtual environment is active.
- For MP3s, install `ffmpeg` and verify `pydub` is installed; otherwise, use WAV.
- If plots are empty in ultrasonic view, the audio might be low sample rate (Nyquist < 18 kHz).

## License
MIT
