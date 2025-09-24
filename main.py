import os
import sys
import io
import datetime
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, get_window

# Optional MP3 support via pydub
try:
	from pydub import AudioSegment
	PYDUB_AVAILABLE = True
except Exception:
	PYDUB_AVAILABLE = False

# Tkinter GUI and matplotlib embedding
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


ANALYSIS_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "analysis_reports")
SAMPLE_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "sample_audio")


@dataclass
class Detection:
	start_time_s: float
	end_time_s: float
	peak_freq_hz: float
	peak_dbfs: float


class AudioWatermarkSnifferApp:
	"""Main application class encapsulating GUI and analysis logic."""

	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("Audio Watermark Sniffer")
		self.root.geometry("1100x750")
		self.root.minsize(900, 600)

		self.sample_rate: Optional[int] = None
		self.audio_mono: Optional[np.ndarray] = None  # float32 in range [-1, 1]
		self.audio_path: Optional[str] = None

		self.only_ultrasonic = tk.BooleanVar(value=True)
		self.sensitivity = tk.DoubleVar(value=6.0)  # dB above median band energy

		self._ensure_directories()
		self._build_gui()

	def _ensure_directories(self) -> None:
		os.makedirs(ANALYSIS_REPORTS_DIR, exist_ok=True)
		os.makedirs(SAMPLE_AUDIO_DIR, exist_ok=True)

	def _build_gui(self) -> None:
		# Top toolbar
		toolbar = ttk.Frame(self.root)
		toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

		load_btn = ttk.Button(toolbar, text="Load Audio", command=self.on_load_audio)
		load_btn.pack(side=tk.LEFT, padx=(0, 8))

		analyze_btn = ttk.Button(toolbar, text="Analyze", command=self.on_analyze)
		analyze_btn.pack(side=tk.LEFT, padx=(0, 8))

		csv_btn = ttk.Button(toolbar, text="Export CSV", command=self.on_export_csv)
		csv_btn.pack(side=tk.LEFT, padx=(0, 8))

		report_btn = ttk.Button(toolbar, text="Save Report", command=self.on_save_report)
		report_btn.pack(side=tk.LEFT, padx=(0, 8))

		# Options
		options_frame = ttk.Frame(toolbar)
		options_frame.pack(side=tk.LEFT, padx=16)

		ultra_chk = ttk.Checkbutton(options_frame, text="Ultrasonic only (>18 kHz)", variable=self.only_ultrasonic, command=self._refresh_plot_view)
		ultra_chk.pack(side=tk.LEFT)

		ttk.Label(options_frame, text="Sensitivity (dB):").pack(side=tk.LEFT, padx=(16, 4))
		sens_scale = ttk.Scale(options_frame, from_=0.0, to=20.0, variable=self.sensitivity, command=lambda _evt: None)
		sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=(0, 8))

		# Main content: left plot, right results
		split = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
		split.pack(fill=tk.BOTH, expand=True)

		# Plot area
		plot_frame = ttk.Frame(split)
		split.add(plot_frame, weight=3)

		self.figure = Figure(figsize=(6, 4), dpi=100)
		self.ax = self.figure.add_subplot(111)
		self.ax.set_title("Frequency Spectrum")
		self.ax.set_xlabel("Frequency (Hz)")
		self.ax.set_ylabel("Magnitude (dBFS)")
		self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

		# Results panel
		right_frame = ttk.Frame(split)
		split.add(right_frame, weight=2)

		results_label = ttk.Label(right_frame, text="Analysis Results")
		results_label.pack(anchor=tk.W, padx=8, pady=(8, 0))

		self.results_text = tk.Text(right_frame, wrap=tk.WORD, height=20)
		self.results_text.configure(state=tk.DISABLED)
		scrollbar = ttk.Scrollbar(right_frame, command=self.results_text.yview)
		self.results_text['yscrollcommand'] = scrollbar.set
		self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

		# Status bar
		self.status = tk.StringVar(value="Load an audio file to begin.")
		statusbar = ttk.Label(self.root, textvariable=self.status, anchor=tk.W)
		statusbar.pack(side=tk.BOTTOM, fill=tk.X)

		self._clear_plot()

	def log_result(self, text: str) -> None:
		self.results_text.configure(state=tk.NORMAL)
		self.results_text.insert(tk.END, text + "\n")
		self.results_text.configure(state=tk.DISABLED)
		self.results_text.see(tk.END)

	def set_status(self, text: str) -> None:
		self.status.set(text)
		self.root.update_idletasks()

	def _clear_plot(self) -> None:
		self.ax.cla()
		self.ax.set_title("Frequency Spectrum")
		self.ax.set_xlabel("Frequency (Hz)")
		self.ax.set_ylabel("Magnitude (dBFS)")
		self.canvas.draw_idle()

	def on_load_audio(self) -> None:
		filetypes = [
			("Audio Files", "*.wav;*.mp3"),
			("WAV", "*.wav"),
			("MP3", "*.mp3"),
			("All Files", "*.*"),
		]
		path = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
		if not path:
			return
		try:
			self._load_audio_file(path)
			self.audio_path = path
			self.set_status(f"Loaded: {os.path.basename(path)} ({self.sample_rate} Hz)")
			self.results_text.configure(state=tk.NORMAL)
			self.results_text.delete("1.0", tk.END)
			self.results_text.configure(state=tk.DISABLED)
			self._plot_overall_spectrum()
		except Exception as exc:
			messagebox.showerror("Load Error", f"Failed to load audio: {exc}")

	def _load_audio_file(self, path: str) -> None:
		ext = os.path.splitext(path)[1].lower()
		if ext == ".wav":
			rate, data = wavfile.read(path)
			mono = self._to_mono_float32(data)
			self.sample_rate = int(rate)
			self.audio_mono = mono
		elif ext == ".mp3":
			if not PYDUB_AVAILABLE:
				raise RuntimeError("pydub not available. Install pydub and ffmpeg for MP3 support.")
			audio_seg = AudioSegment.from_file(path, format="mp3")
			audio_seg = audio_seg.set_channels(1)
			audio_seg = audio_seg.set_frame_rate(audio_seg.frame_rate)
			# Convert to numpy float32 [-1,1]
			samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
			if audio_seg.sample_width == 2:
				samples /= 32768.0
			elif audio_seg.sample_width == 4:
				samples /= 2147483648.0
			else:
				samples /= (2 ** (8 * audio_seg.sample_width - 1))
			self.sample_rate = int(audio_seg.frame_rate)
			self.audio_mono = samples
		else:
			raise RuntimeError("Unsupported file format. Please select WAV or MP3.")

	def _to_mono_float32(self, data: np.ndarray) -> np.ndarray:
		# Convert to mono if stereo
		if data.ndim == 2 and data.shape[1] > 1:
			data = data.mean(axis=1)
		# Handle common integer PCM types first (keep dtype until scaling)
		if data.dtype == np.int16:
			return (data.astype(np.float32) / 32768.0)
		if data.dtype == np.int32:
			return (data.astype(np.float32) / 2147483648.0)
		if data.dtype == np.uint8:
			# Unsigned 8-bit PCM: 128 is zero
			return ((data.astype(np.float32) - 128.0) / 128.0)
		# For float types, ensure float32 and normalize if outside [-1, 1]
		float_data = data.astype(np.float32)
		max_abs = float(np.max(np.abs(float_data))) if float_data.size else 1.0
		if max_abs > 1.0 and max_abs > 0:
			float_data = float_data / max_abs
		return float_data

	def _plot_overall_spectrum(self) -> None:
		if self.audio_mono is None or self.sample_rate is None:
			return
		n = len(self.audio_mono)
		win = get_window("hann", n, fftbins=True)
		sig = self.audio_mono * win
		fft_vals = np.fft.rfft(sig, n=n)
		freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
		mag = np.abs(fft_vals) + 1e-12
		dbfs = 20.0 * np.log10(mag / np.max(mag))

		self.ax.cla()
		self.ax.plot(freqs, dbfs, color="#1f77b4", linewidth=0.8)
		self.ax.set_title("Frequency Spectrum (Full)" if not self.only_ultrasonic.get() else "Frequency Spectrum (Ultrasonic View)")
		self.ax.set_xlabel("Frequency (Hz)")
		self.ax.set_ylabel("Magnitude (dBFS)")
		self.ax.grid(True, which="both", linestyle=":", alpha=0.4)

		self._apply_xlimits()
		self.canvas.draw_idle()

	def _apply_xlimits(self) -> None:
		if self.sample_rate is None:
			return
		if self.only_ultrasonic.get():
			self.ax.set_xlim(18000, self.sample_rate / 2)
		else:
			self.ax.set_xlim(0, self.sample_rate / 2)

	def _refresh_plot_view(self) -> None:
		# Re-plot if we have audio; otherwise just adjust limits
		if self.audio_mono is not None:
			self._plot_overall_spectrum()
		else:
			self._clear_plot()

	def on_analyze(self) -> None:
		if self.audio_mono is None or self.sample_rate is None:
			messagebox.showinfo("No Audio", "Please load an audio file first.")
			return

		# Run heavy analysis in a thread to keep UI responsive
		threading.Thread(target=self._run_analysis_thread, daemon=True).start()

	def _run_analysis_thread(self) -> None:
		try:
			self.set_status("Analyzing... this may take a moment")
			detections = self._detect_ultrasonic_regions(
				signal=self.audio_mono,
				sample_rate=self.sample_rate,
				ultrasonic_hz=18000.0,
				sensitivity_db=float(self.sensitivity.get()),
			)
			self._display_detections_and_highlight(detections)
			self.set_status(f"Analysis complete. Found {len(detections)} region(s).")
		except Exception as exc:
			self.set_status("Analysis failed")
			messagebox.showerror("Analysis Error", str(exc))

	def _detect_ultrasonic_regions(
		self,
		signal: np.ndarray,
		sample_rate: int,
		ultrasonic_hz: float = 18000.0,
		sensitivity_db: float = 6.0,
		window_s: float = 0.1,
		overlap: float = 0.5,
	) -> List[Detection]:
		"""Detect regions where high-frequency energy exceeds a dynamic threshold.

		Algorithm overview:
		- Compute STFT with Hann window.
		- For each time frame, compute band energy above ultrasonic_hz.
		- Compute median energy across time as baseline; flag frames exceeding baseline + sensitivity_db.
		- Merge consecutive flagged frames into continuous regions and record their peak frequency and level.
		"""
		nperseg = max(256, int(window_s * sample_rate))
		noverlap = int(nperseg * overlap)
		win = get_window("hann", nperseg, fftbins=True)

		freqs, times, Zxx = stft(signal, fs=sample_rate, window=win, nperseg=nperseg, noverlap=noverlap, boundary=None)
		magnitude = np.abs(Zxx) + 1e-12
		# Convert to dBFS relative to per-frame max to be robust
		per_frame_max = np.maximum(np.max(magnitude, axis=0, keepdims=True), 1e-12)
		dbfs = 20.0 * np.log10(magnitude / per_frame_max)

		ultra_idx = np.where(freqs >= ultrasonic_hz)[0]
		if ultra_idx.size == 0:
			return []

		ultra_dbfs = dbfs[ultra_idx, :]  # shape: [F, T]
		# Aggregate energy across ultrasonic band per frame using median for robustness
		band_energy_db = np.median(ultra_dbfs, axis=0)

		baseline = np.median(band_energy_db)
		threshold = baseline + sensitivity_db

		flagged = band_energy_db > threshold

		detections: List[Detection] = []
		current_start: Optional[int] = None
		for idx, is_flagged in enumerate(flagged):
			if is_flagged and current_start is None:
				current_start = idx
			elif not is_flagged and current_start is not None:
				start_idx = current_start
				end_idx = idx - 1
				det = self._summarize_region(freqs, times, dbfs, ultra_idx, start_idx, end_idx)
				detections.append(det)
				current_start = None
		# Tail
		if current_start is not None:
			start_idx = current_start
			end_idx = len(times) - 1
			detections.append(self._summarize_region(freqs, times, dbfs, ultra_idx, start_idx, end_idx))

		return detections

	def _summarize_region(
		self,
		freqs: np.ndarray,
		times: np.ndarray,
		dbfs: np.ndarray,
		ultra_idx: np.ndarray,
		start_idx: int,
		end_idx: int,
	) -> Detection:
		region_db = dbfs[:, start_idx:end_idx + 1]
		# Focus only ultrasonic rows
		region_ultra_db = region_db[ultra_idx, :]
		# Find peak within region
		peak_flat_idx = np.argmax(region_ultra_db)
		peak_freq_row, peak_time_col = np.unravel_index(peak_flat_idx, region_ultra_db.shape)
		peak_freq = freqs[ultra_idx[peak_freq_row]]
		peak_db = float(region_ultra_db[peak_freq_row, peak_time_col])
		start_time = float(times[start_idx])
		end_time = float(times[end_idx])
		return Detection(start_time_s=start_time, end_time_s=end_time, peak_freq_hz=float(peak_freq), peak_dbfs=peak_db)

	def _display_detections_and_highlight(self, detections: List[Detection]) -> None:
		# Log results
		self.results_text.configure(state=tk.NORMAL)
		self.results_text.delete("1.0", tk.END)
		for i, d in enumerate(detections, start=1):
			self.results_text.insert(
				tk.END,
				f"[{i}] {d.start_time_s:.2f}s - {d.end_time_s:.2f}s | Peak: {d.peak_freq_hz/1000:.2f} kHz @ {d.peak_dbfs:.1f} dB\n",
			)
		if not detections:
			self.results_text.insert(tk.END, "No ultrasonic regions detected above the adaptive threshold.\n")
		self.results_text.configure(state=tk.DISABLED)
		self.results_text.see(tk.END)

		# Update plot with highlights: overall spectrum + mark ultrasonic band mean level
		self._plot_overall_spectrum()
		if self.audio_mono is None or self.sample_rate is None:
			return

		# Overlay highlighted peaks on the spectrum by marking vertical lines at peak freqs
		if detections:
			peak_freqs = [d.peak_freq_hz for d in detections]
			for pf in peak_freqs:
				self.ax.axvline(pf, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
			self.canvas.draw_idle()

	def on_save_report(self) -> None:
		if self.audio_path is None:
			messagebox.showinfo("No Audio", "Please load an audio file first.")
			return
		# Collect current text panel content
		text = self.results_text.get("1.0", tk.END).strip()
		if not text:
			messagebox.showinfo("No Results", "Please run analysis first.")
			return
		basename = os.path.splitext(os.path.basename(self.audio_path))[0]
		stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		fname = f"report_{basename}_{stamp}.txt"
		fpath = os.path.join(ANALYSIS_REPORTS_DIR, fname)
		with open(fpath, "w", encoding="utf-8") as f:
			f.write(f"Audio Watermark Sniffer Report\n")
			f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
			f.write(f"File: {self.audio_path}\n")
			f.write(f"Sample Rate: {self.sample_rate} Hz\n")
			f.write(f"Ultrasonic only: {self.only_ultrasonic.get()}\n")
			f.write(f"Sensitivity (dB): {self.sensitivity.get():.1f}\n")
			f.write("\nDetections:\n")
			f.write(text + "\n")
		messagebox.showinfo("Report Saved", f"Saved to:\n{fpath}")

	def on_export_csv(self) -> None:
		if self.audio_path is None:
			messagebox.showinfo("No Audio", "Please load an audio file first.")
			return
		text = self.results_text.get("1.0", tk.END).strip()
		if not text or text.startswith("No ultrasonic"):
			messagebox.showinfo("No Detections", "Run analysis with detections before exporting.")
			return
		# Parse the results panel lines
		lines = [ln for ln in text.splitlines() if ln.strip().startswith("[")]
		rows: List[Tuple[float, float, float, float]] = []
		for ln in lines:
			try:
				# Format: [i] start - end | Peak: X kHz @ Y dB
				parts = ln.split("]")[-1].strip()
				time_part, peak_part = parts.split("|")
				start_s = float(time_part.split("-")[0].strip().rstrip("s"))
				end_s = float(time_part.split("-")[1].strip().rstrip("s"))
				peak_freq_khz = float(peak_part.split("Peak:")[-1].split("kHz")[0].strip())
				peak_db = float(peak_part.split("@")[1].split("dB")[0].strip())
				rows.append((start_s, end_s, peak_freq_khz * 1000.0, peak_db))
			except Exception:
				continue
		if not rows:
			messagebox.showinfo("No Detections", "Could not parse any detections to export.")
			return
		basename = os.path.splitext(os.path.basename(self.audio_path))[0]
		stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		fname = f"detections_{basename}_{stamp}.csv"
		fpath = os.path.join(ANALYSIS_REPORTS_DIR, fname)
		with open(fpath, "w", encoding="utf-8") as f:
			f.write("start_time_s,end_time_s,peak_freq_hz,peak_dbfs\n")
			for r in rows:
				f.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.2f},{r[3]:.2f}\n")
		messagebox.showinfo("CSV Exported", f"Saved to:\n{fpath}")


def main() -> None:
	root = tk.Tk()
	# Use themed widgets
	try:
		style = ttk.Style()
		if sys.platform == "win32":
			style.theme_use("vista")
		else:
			style.theme_use(style.theme_use())
	except Exception:
		pass
	app = AudioWatermarkSnifferApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
