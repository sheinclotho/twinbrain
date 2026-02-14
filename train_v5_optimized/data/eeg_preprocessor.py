import mne
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

class EEGPreprocessor:
    def __init__(self,
                 l_freq: float = 1.0,
                 h_freq: float = 40.0,
                 resample_sfreq: float = 250.0,
                 use_ica: bool = True,
                 drop_non_eeg: bool = False):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resample_sfreq = resample_sfreq
        self.use_ica = use_ica
        self.drop_non_eeg = drop_non_eeg

    def _select_file(self, title: str, filetypes: list):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        if not file_path:
            raise FileNotFoundError(f"No file selected for {title}")
        return file_path

    def load_raw(self, eeg_file: str = None):
        if eeg_file is None:
            eeg_file = self._select_file(
                title="请选择 EEG 文件",
                filetypes=[("EEG files", "*.fif *.edf *.set"), ("All files", "*.*")]
            )
        else:
            eeg_file = str(eeg_file)

        if not os.path.exists(eeg_file):
            raise FileNotFoundError(f"[EEG] 文件不存在: {eeg_file}")

        try:
            if eeg_file.endswith(".fif"):
                raw = mne.io.read_raw_fif(eeg_file, preload=True)
            elif eeg_file.endswith(".edf"):
                raw = mne.io.read_raw_edf(eeg_file, preload=True)
            elif eeg_file.endswith(".set"):
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
            else:
                raise ValueError(f"Unsupported format: {eeg_file}")
            return raw
        except Exception as e:
            print(f"[ERROR] MNE 读取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess(self, eeg_file: str = None, montage: str = "standard_1020", manual_ica_exclude: list = None):
        try:
            raw = self.load_raw(eeg_file)
            # 不要用 raw.filename！用 eeg_file 代替
            print(f"[EEG] 成功加载文件: {eeg_file}")
        except Exception as e:
            print(f"[ERROR] 加载 EEG 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        try:
            # 非 EEG 通道处理
            ch_types = raw.get_channel_types()
            mapping = {}
            for ch_name, ch_type in zip(raw.ch_names, ch_types):
                if ch_type.upper() in ["ECG", "EOG", "EMG"]:
                    mapping[ch_name] = ch_type.lower()
            if mapping:
                raw.set_channel_types(mapping)
                print(f"[EEG] 设置非 EEG 通道类型: {list(mapping.keys())}")

            # Montage
            if montage is not None:
                raw.set_montage(montage, on_missing="ignore")
                print(f"[EEG] 设置 Montage: {montage}")

            # 滤波 + 平均参考
            print(f"[EEG] 滤波: {self.l_freq}–{self.h_freq} Hz")
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)
            raw.set_eeg_reference("average", projection=True)
            raw.apply_proj()

            # ICA
            if self.use_ica:
                print("[EEG] 运行 ICA 去伪迹...")
                n_comp = min(20, len(raw.ch_names) - 1)
                ica = mne.preprocessing.ICA(n_components=n_comp, random_state=97, max_iter="auto")
                ica.fit(raw)
                eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
                if eog_picks.size > 0:
                    eog_indices, _ = ica.find_bads_eog(raw)
                    ica.exclude.extend(eog_indices)
                    print(f"[EEG] ICA 自动排除 EOG 成分: {eog_indices}")
                if manual_ica_exclude:
                    ica.exclude.extend(manual_ica_exclude)
                    print(f"[EEG] 手动排除 ICA 成分: {manual_ica_exclude}")
                raw = ica.apply(raw.copy())

            # 删除 ECG 通道
            if "ECG" in raw.ch_names:
                raw.drop_channels(["ECG"])
                print("[EEG] 删除 ECG 通道")

            # 重采样
            if self.resample_sfreq is not None:
                print(f"[EEG] 重采样到 {self.resample_sfreq} Hz")
                raw.resample(self.resample_sfreq)

            print(f"[EEG] 预处理完成: {len(raw.ch_names)} 通道, {raw.n_times} 采样点")
            return raw

        except Exception as e:
            print(f"[ERROR] EEG 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _smooth_spikes(self, data: np.ndarray, percentile: float = 99.0, taper_pct: float = 0.05) -> np.ndarray:
        """
        修正 EEG 信号头尾或局部 spike，保证归一化不被极端值拉扯。
        data: (n_epochs, n_channels, n_times) 或 (n_channels, n_times)
        percentile: 用于中间 spike 裁剪
        taper_pct: 头尾 taper 比例（Hann 窗平滑）
        """
        orig_shape = data.shape
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # (1, ch, t)

        n_epochs, n_ch, n_times = data.shape
        n_taper = int(n_times * taper_pct)

        # 1) taper 头尾
        if n_taper > 0:
            window = np.hanning(n_taper * 2)
            fade_in = window[:n_taper]
            fade_out = window[-n_taper:]
            for ep in range(n_epochs):
                for ch in range(n_ch):
                    data[ep, ch, :n_taper] *= fade_in
                    data[ep, ch, -n_taper:] *= fade_out

        # 2) 中间 spike 裁剪 + 插值
        for ep in range(n_epochs):
            for ch in range(n_ch):
                x = data[ep, ch]
                threshold = np.percentile(np.abs(x), percentile)
                spike_idx = np.where(np.abs(x) > threshold)[0]
                if spike_idx.size > 0:
                    valid_idx = np.setdiff1d(np.arange(n_times), spike_idx)
                    if valid_idx.size > 1:  # 避免全 spike
                        x[spike_idx] = np.interp(spike_idx, valid_idx, x[valid_idx])
                    data[ep, ch] = x

        if orig_shape[0] == n_ch and len(orig_shape) == 2:
            return data[0]
        return data
        
    def extract_epochs(self, raw, epoch_length: float = 2.0, smooth_spike: bool = True):
        events = mne.make_fixed_length_events(raw, duration=epoch_length)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length,
                            baseline=None, preload=True)
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        if smooth_spike:
            data = self._smooth_spikes(data, percentile=99.0, taper_pct=0.05)

        return data


    def compute_source_localization(self, raw, fwd_file: str = None, noise_cov_file: str = None, method: str = "dSPM"):
        if fwd_file is None:
            fwd_file = self._select_file(
                title="请选择 Forward Model 文件 (.fif)",
                filetypes=[("Forward model", "*.fif"), ("All files", "*.*")]
            )
        fwd = mne.read_forward_solution(fwd_file)

        if noise_cov_file is None:
            noise_cov = mne.compute_raw_covariance(raw)
        else:
            if noise_cov_file == "ASK":
                noise_cov_file = self._select_file(
                    title="请选择 Noise Covariance 文件 (.fif)",
                    filetypes=[("Noise cov", "*.fif"), ("All files", "*.*")]
                )
            noise_cov = mne.read_cov(noise_cov_file)

        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
        stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1. / 9., method=method)
        return stc
