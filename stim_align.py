# stim_align.py
import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import mne
import torch

# ==============================================
# 工具函数
# ==============================================

def stim_list_to_dict(stim_all, node_types=("eeg", "fmri")):
    """
    将 batch_generate_stim 返回的 stim list 转为字典形式，
    每个任务独立 tensor，避免尺寸不匹配。
    
    返回格式：
    stim_dict = {
        "eeg": { "TASK1": tensor1, "TASK2": tensor2, ... },
        "fmri": { "TASK1": tensor1, ... }
    }
    """
    stim_dict = {ntype: {} for ntype in node_types}

    for df in stim_all:
        # 从文件名或 Onset/TR 列获取 task 名（假设文件名里有 task 信息）
        if "task" in df.columns:
            task_name = df["task"].iloc[0].upper()
        else:
            # 如果没有 task 列，则从文件名里提取
            # 假设 df 有来源文件名列 "source_file"
            task_name = df.get("source_file", ["UNKNOWN"])[0].upper()

        for ntype in node_types:
            if ntype.upper() in task_name or ntype.lower() in df.columns:
                # 取对应列，如果没有，就全部数值作为 tensor
                cols = [c for c in df.columns if c.lower().startswith(ntype)]
                if not cols:
                    cols = [c for c in df.columns if c not in ("Trial", "Onset", "fMRI_TR", "task", "source_file")]
                data = df[cols].values.astype(np.float32)
                stim_tensor = torch.from_numpy(data)
                stim_dict[ntype][task_name] = stim_tensor

    return stim_dict


def discover_eeg_tasks(eeg_dir: Path):
    eeg_dir = Path(eeg_dir)
    files = list(eeg_dir.glob("sub-*_task-*_run-*_eeg.*"))
    tasks = set()
    for f in files:
        m = re.search(r"task-([A-Za-z]+)", f.name)
        if m:
            token = m.group(1).upper()
            if token.endswith("ON"):
                base = token[:-2]
            elif token.endswith("OFF"):
                base = token[:-3]
            else:
                base = token
            tasks.add(base)
    return sorted(list(tasks))

def discover_fmri_tasks(func_dir: Path):
    func_dir = Path(func_dir)
    files = list(func_dir.glob("sub-*_task-*_run-*_bold.nii*"))
    tasks = set()
    for f in files:
        m = re.search(r"task-([A-Za-z0-9]+)", f.name)
        if m:
            tasks.add(m.group(1).upper())
    return sorted(list(tasks))

def get_tr_from_fmri(func_dir: Path, task: str = None, default_tr: float = 2.0) -> float:
    """
    自动从 fMRI NIfTI 文件中读取 TR。若 task 未给出，则读取第一个文件。
    """
    func_dir = Path(func_dir)
    if task:
        fmri_files = list(func_dir.glob(f"*task-{task.lower()}*_bold.nii*"))
    else:
        fmri_files = list(func_dir.glob("*_bold.nii*"))
    if not fmri_files:
        return default_tr
    try:
        img = nib.load(str(fmri_files[0]))
        hdr = img.header
        tr = hdr.get("pixdim", [0, 0, 0, default_tr])[4]
        if tr <= 0 and "RepetitionTime" in hdr:
            tr = hdr["RepetitionTime"]
        return float(tr)
    except Exception as e:
        print(f"[Warn] TR parse failed ({task}): {e}")
        return default_tr

# ==============================================
# Stim 构建模块
# ==============================================
def load_behavior(beh_file: Path):
    df = pd.read_csv(beh_file, sep="\t")
    cols = ['Trial', 'ImgType', 'ImgNum', 'Onset', 'RT', 'PressedC']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

def eeg_onset_to_samples(eeg_file: Path):
    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=False)
    events, event_id = mne.events_from_annotations(raw)
    sfreq = raw.info['sfreq']
    out = []
    for e in events:
        out.append({'EEG_sample': e[0], 'EEG_event_code': e[2]})
    return pd.DataFrame(out), sfreq

def build_stim_for_task(eeg_dir: Path, beh_dir: Path, task: str, stim_dir: Path, fmri_tr: float):
    """
    生成单任务 stim 文件。
    若 beh 文件存在 -> 使用行为数据；
    否则仅基于 EEG 事件生成。
    """
    stim_dir.mkdir(exist_ok=True, parents=True)
    eeg_files = list(Path(eeg_dir).glob(f"*task-{task.lower()}*_eeg.set"))

    # 尝试行为文件
    beh_candidates = list(Path(beh_dir).glob(f"*{task.lower()}*.tsv")) if beh_dir else []
    beh_df = None
    if beh_candidates:
        try:
            beh_df = load_behavior(beh_candidates[0])
            print(f"[Stim] Using behavior file for {task}: {beh_candidates[0].name}")
        except Exception as e:
            print(f"[Warn] Failed to load behavior for {task}: {e}")

    eeg_stims = []
    for eeg_file in eeg_files:
        eeg_df, sfreq = eeg_onset_to_samples(eeg_file)
        if beh_df is not None:
            beh_df = beh_df.copy()
            beh_df["EEG_sample"] = (beh_df["Onset"] * sfreq).astype(int)
        else:
            # 无行为文件时，仅用 EEG 事件时间
            eeg_df["Onset"] = eeg_df["EEG_sample"] / sfreq
            beh_df = eeg_df.copy()
            beh_df["Trial"] = np.arange(len(beh_df))
            beh_df["ImgType"] = "Unknown"
            beh_df["ImgNum"] = np.nan
            beh_df["RT"] = np.nan
            beh_df["PressedC"] = np.nan
        beh_df["fMRI_TR"] = (beh_df["Onset"] / fmri_tr).astype(int)
        eeg_stims.append(beh_df)

    stim_df = pd.concat(eeg_stims, ignore_index=True) if eeg_stims else pd.DataFrame()
    out_path = stim_dir / f"stim_{task}.tsv"
    stim_df.to_csv(out_path, sep="\t", index=False)
    print(f"[Stim] Saved {task} stim -> {out_path}")
    return stim_df

# ==============================================
# 主函数
# ==============================================
def batch_generate_stim(bids_root: Path, target_task: str = None):
    """
    自动批量生成 stim 文件。
    1. 扫描 EEG 任务；
    2. 检查 stim 是否存在；
    3. 若不存在则生成；
    4. 自动从 fMRI 文件读取 TR。
    """
    eeg_dir = Path(bids_root) / "eeg"
    func_dir = Path(bids_root) / "func"
    beh_dir = Path(bids_root) / "beh"
    stim_dir = Path(bids_root) / "stim"

    eeg_tasks = discover_eeg_tasks(eeg_dir)
    fmri_tasks = discover_fmri_tasks(func_dir)
    all_tasks = [target_task.upper()] if target_task else sorted(set(eeg_tasks) | set(fmri_tasks))

    stim_all = []
    for task in all_tasks:
        stim_file = stim_dir / f"stim_{task}.tsv"
        if stim_file.exists():
            print(f"[Stim] Found existing file for {task}, skip generation.")
            stim_all.append(pd.read_csv(stim_file, sep="\t"))
            continue

        fmri_tr = get_tr_from_fmri(func_dir, task)
        print(f"[TR] Using TR={fmri_tr:.3f}s for task={task}")
        stim_df = build_stim_for_task(eeg_dir, beh_dir, task, stim_dir, fmri_tr)
        stim_all.append(stim_df)

    print(f"[Batch] Generated {len(stim_all)} stim tables: {all_tasks}")
    stim = stim_list_to_dict(stim_all)
    return stim


if __name__ == "__main__":
    bids_root = Path("/path/to/BIDS")
    stim_list = batch_generate_stim(bids_root)
