import numpy as np
import pandas as pd
from nilearn import image
from nilearn.input_data import NiftiMasker
from tqdm import tqdm


class FMRI_Preprocessor:
    def __init__(self, tr=2.0, high_pass=0.01, low_pass=0.1, smoothing_fwhm=6.0):
        self.masker = NiftiMasker(
            standardize=True,
            detrend=True,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
            smoothing_fwhm=smoothing_fwhm
        )
        self.ts_ = None
        self.fmri_img_ = None
        self.confounds_ = None
        self.tr=tr

    def preprocess(self, fmri_file, confounds_file=None):
        steps = [
            "读取 fMRI NIfTI 文件",
            "读取混杂变量文件" if confounds_file is not None else "跳过混杂变量",
            "应用 NiftiMasker 进行预处理"
        ]
        with tqdm(total=len(steps), desc="fMRI Preprocessing", ncols=100) as pbar:
            # Step 1
            self.fmri_img_ = image.load_img(fmri_file)
            pbar.set_postfix_str("fMRI 加载完成")
            pbar.update(1)

            # Step 2
            confounds = None
            if confounds_file is not None:
                if confounds_file.endswith(".tsv"):
                    confounds = pd.read_csv(confounds_file, sep="\t")
                elif confounds_file.endswith(".csv"):
                    confounds = pd.read_csv(confounds_file)
                else:
                    raise ValueError("Confounds file must be .tsv or .csv")
                confounds = confounds.fillna(0)
                self.confounds_ = confounds
                pbar.set_postfix_str("混杂变量加载完成")
            else:
                pbar.set_postfix_str("未提供混杂变量")
            pbar.update(1)

            # Step 3
            ts = self.masker.fit_transform(self.fmri_img_, confounds=confounds)
            self.ts_ = ts.astype(np.float32)
            
            # 关键修复：生成 clean_img
            self.clean_img = self.inverse_transform(self.ts_)
            
            pbar.set_postfix_str("预处理完成 + clean_img 生成")
            pbar.update(1)
        
        return self.ts_
        
    def get_time_series(self):
        if self.ts_ is None:
            raise ValueError("请先调用 preprocess() 进行预处理")
        return self.ts_

    def inverse_transform(self, ts):
        with tqdm(total=1, desc="🔄 Inverse Transform", ncols=100) as pbar:
            img = self.masker.inverse_transform(ts)
            pbar.set_postfix_str("NIfTI 生成完成")
            pbar.update(1)
        return img

    def save_clean_img(self, save_path):
        if self.ts_ is None:
            raise ValueError("请先调用 preprocess()")

        with tqdm(total=2, desc="💾 Save Clean fMRI", ncols=100) as pbar:
            clean_img = self.inverse_transform(self.ts_)
            pbar.set_postfix_str("NIfTI 转换完成")
            pbar.update(1)

            clean_img.to_filename(save_path)
            pbar.set_postfix_str("保存完成")
            pbar.update(1)

        print(f"🎉 Preprocessed fMRI image saved to {save_path}")
