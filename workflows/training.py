"""
Training workflow for TwinBrain.
Handles the complete training pipeline from data loading to model training.
"""

import os
import torch
from pathlib import Path
from typing import Optional, List, Dict
import logging

from utils.config import Config
from utils.logging_utils import log_stage, get_logger
from utils.analysis import compute_xcorr_best_lag
from utils.utils import set_random_seed
from mapper.atlas_mapper import BrainAtlas
from train.hetero_trainer import DynamicHeteroTrainer
from utils.function import (
    discover_eeg_tasks,
    discover_fmri_tasks,
    load_fmri,
    load_dti,
    load_eeg,
    load_atlas,
    build_nodes,
    save_nodes_json,
    build_hetero_graph
)

# Optional diagnostic imports
try:
    from utils.diagnostics import run_comprehensive_diagnostics
    # Keep legacy imports for backward compatibility
    from utils.debug import run_decoder_only_warmup
except ImportError:
    run_comprehensive_diagnostics = None
    run_decoder_only_warmup = None

logger = get_logger(__name__)


class TrainingWorkflow:
    """Complete training workflow for TwinBrain."""
    
    def __init__(self, config: Config, base_dir: Path):
        """
        Initialize training workflow.
        
        Args:
            config: Configuration object
            base_dir: Base directory containing subject data
        """
        self.config = config
        self.base_dir = Path(base_dir)
        self.subjects = [d for d in self.base_dir.glob("sub-*") if d.is_dir()]
        
        if not self.subjects:
            raise ValueError(f"No subjects found in {self.base_dir}")
        
        # Initialize random seeds for reproducibility and CUDA safety
        # This prevents THPGenerator_initDefaultGenerator errors
        seed = self.config.get('random_seed', 42)
        set_random_seed(seed)
        logger.info(f"Random seed set to {seed} in TrainingWorkflow")
        
        logger.info(f"Found {len(self.subjects)} subjects in {self.base_dir}")
    
    def _setup_paths(self, subject_dir: Path) -> Dict[str, Path]:
        """Setup paths for a subject."""
        result_dir = subject_dir / self.config.get('output.results_dir', 'results')
        result_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {
            "eeg_dir": subject_dir / "eeg",
            "func_dir": subject_dir / "func",
            "dti_npy": subject_dir / "dwi" / f"{subject_dir.name}_acq-AP_dwi_connectome.npy",
            "nodes_json": result_dir / "nodes.json",
            "hetero_model": result_dir / "hetero_gnn_trained.pt",
            "result_dir": result_dir,
        }
        
        # Create parent directories
        for path in paths.values():
            if path.parent != subject_dir:
                path.parent.mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def _load_or_generate_data(self, subject_dir: Path, paths: Dict[str, Path], atlas: BrainAtlas):
        """Load cached data or generate from scratch."""
        result_dir = paths["result_dir"]
        
        # Cache file paths
        cache_dir = result_dir / self.config.get('data.cache_dir', 'cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        eeg_data_cache = cache_dir / "eeg_data.pt"
        hetero_graphs_cache = cache_dir / "hetero_graphs.pt"
        
        use_cache = self.config.get('data.use_cache', True)
        
        # Check cache
        if use_cache and eeg_data_cache.exists() and hetero_graphs_cache.exists():
            logger.info("Loading cached preprocessed data")
            eeg_data = torch.load(eeg_data_cache, map_location="cpu", weights_only=False)
            hetero_graphs = torch.load(hetero_graphs_cache, map_location="cpu", weights_only=False)
        else:
            logger.info("Preprocessing data (cache miss or disabled)")
            
            # Discover tasks
            eeg_tasks = discover_eeg_tasks(paths["eeg_dir"])
            fmri_tasks = discover_fmri_tasks(paths["func_dir"])
            logger.info(f"EEG tasks: {eeg_tasks}")
            logger.info(f"fMRI tasks: {fmri_tasks}")
            
            # Load atlas files
            atlas_file = self.base_dir.parent / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii"
            label_file = self.base_dir.parent / "schaefer200_mask_ready.json"
            
            # Load fMRI data
            with log_stage("fMRI Data Loading"):
                fmri_data = load_fmri(
                    func_dir=paths["func_dir"],
                    tasks=fmri_tasks,
                    atlas_file=atlas_file,
                    label_file=label_file,
                    brain_atlas=atlas,
                    output_root=result_dir
                )
            
            # Load EEG data
            with log_stage("EEG Data Loading"):
                eeg_data = load_eeg(
                    eeg_dir=paths["eeg_dir"],
                    brain_atlas=atlas,
                    output_root=result_dir
                )
                if use_cache:
                    torch.save(eeg_data, eeg_data_cache)
            
            # Load DTI and build nodes
            with log_stage("Graph Construction"):
                dti = load_dti(paths["dti_npy"])
                nodes = build_nodes(atlas)
                save_nodes_json(nodes, paths["nodes_json"])
                
                # Build hetero graph without stimulus data (stim_dict is optional)
                hetero_graphs = build_hetero_graph(fmri_data, eeg_data, stim_dict=None)
                if use_cache:
                    torch.save(hetero_graphs, hetero_graphs_cache)
        
        return eeg_data, hetero_graphs
    
    def _create_trainer(self, hetero_graphs, result_dir: Path) -> DynamicHeteroTrainer:
        """Create and configure trainer from config."""
        
        # Extract configuration
        cfg = self.config
        
        trainer = DynamicHeteroTrainer(
            hetero_data=hetero_graphs,
            hidden_dim=cfg.get('model.hidden_dim', 128),
            num_epochs=100,  # Default; actual epochs specified in train() calls
            recon_weight=cfg.get('loss.recon_weight', 1.0),
            recon_norm_weight=cfg.get('loss.recon_norm_weight', 3.0),
            recon_corr_weight=cfg.get('loss.recon_corr_weight', 2.0),
            recon_feat_var_weight=cfg.get('loss.recon_feat_var_weight', 0.02),
            temp_weight=cfg.get('loss.temp_weight', 2.0),  # Will be updated during fine-tuning
            feature_lr_mul=cfg.get('training.feature_lr_mul', 12.0),
            scale_lr_mul=cfg.get('training.scale_lr_mul', 10.0),
            warmup_epochs=cfg.get('training.warmup_epochs', 5),
            # New: prediction parameters
            enable_prediction=cfg.get('prediction.enabled', False),
            prediction_context_length=cfg.get('prediction.context_length', None),
            prediction_steps=cfg.get('prediction.steps', 10),
            prediction_weight=cfg.get('prediction.weight', 0.1),
            # New: metrics tracking
            enable_metrics_tracking=cfg.get('metrics.enabled', True),
            metrics_output_dir=str(result_dir / cfg.get('metrics.output_dir', 'metrics')),
            # New: gradient accumulation
            gradient_accumulation_steps=cfg.get('training.gradient_accumulation_steps', 1),
            # Note: device is automatically detected in DynamicHeteroTrainer.__init__
        )
        
        # Configure diagnostics
        if cfg.get('diagnostics.enabled', True):
            trainer.diagnostic_dir = str(result_dir / cfg.get('diagnostics.diagnostic_dir', 'diagnostics'))
            os.makedirs(trainer.diagnostic_dir, exist_ok=True)
        
        # Configure alignment
        trainer.auto_align = cfg.get('alignment.auto_align', True)
        trainer.auto_align_max_lag = cfg.get('alignment.auto_align_max_lag', 150)
        
        return trainer
    
    def _run_diagnostics(self, trainer: DynamicHeteroTrainer):
        """Run comprehensive diagnostic checks."""
        if not self.config.get('diagnostics.enabled', True):
            return
        
        logger.info("Running diagnostics")
        
        # Maximum nodes to analyze for diagnostics
        MAX_DIAGNOSTIC_NODES = 3
        
        try:
            if run_comprehensive_diagnostics is not None:
                save_plots = self.config.get('diagnostics.save_plots', True)
                plot_nodes = self.config.get('diagnostics.plot_nodes', [0, 1, 2])[:MAX_DIAGNOSTIC_NODES]
                
                diag_result = run_comprehensive_diagnostics(
                    trainer,
                    save_dir=trainer.diagnostic_dir,
                    save_plots=save_plots,
                    plot_nodes=plot_nodes,
                )
                
                if "error" not in diag_result:
                    logger.info(f"Diagnostics completed for modalities: {list(diag_result.get('modalities', {}).keys())}")
                    if "summary_file" in diag_result:
                        logger.info(f"Diagnostic summary saved to {diag_result['summary_file']}")
                else:
                    logger.warning(f"Diagnostics failed: {diag_result.get('error', 'unknown')}")
        except Exception as e:
            logger.warning(f"Diagnostics failed: {e}")
    
    def train_subject(self, subject_dir: Path, atlas: BrainAtlas):
        """
        Train model for a single subject with proper training stages.
        
        Training consists of three stages:
        1. Warmup: Initialize model with frozen scale parameters
        2. Main Training: Primary training with default loss weights
        3. Fine-tuning: Refine with stronger temporal alignment
        
        Args:
            subject_dir: Subject directory path
            atlas: Brain atlas
        """
        logger.info(f"Processing subject: {subject_dir.name}")
        
        # Setup paths
        paths = self._setup_paths(subject_dir)
        result_dir = paths["result_dir"]
        
        # Load/generate data
        with log_stage("Data Loading"):
            eeg_data, hetero_graphs = self._load_or_generate_data(
                subject_dir, paths, atlas
            )
        
        # Create trainer
        with log_stage("Trainer Initialization"):
            trainer = self._create_trainer(hetero_graphs, result_dir)
        
        # Initial diagnostics
        self._run_diagnostics(trainer)
        
        # ============================================================
        # Stage 1: Warmup - Initialize with frozen scale parameters
        # ============================================================
        warmup_epochs = self.config.get('training.warmup_epochs', 5)
        
        if warmup_epochs > 0:
            # Optionally use lower learning rate for warmup
            warmup_lr = self.config.get('training.warmup_learning_rate', None)
            original_lr = None
            if warmup_lr is not None:
                original_lr = trainer.optimizer.param_groups[0]['lr']
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup: Using reduced learning rate {warmup_lr}")
            
            with log_stage(f"Warmup Stage ({warmup_epochs} epochs, scale frozen)"):
                trainer.train(num_epochs=warmup_epochs, verbose=True)
            
            # Restore original learning rate if it was changed
            if original_lr is not None:
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = original_lr
                logger.info(f"Warmup complete: Restored learning rate to {original_lr}")
        else:
            logger.info("Warmup stage skipped (warmup_epochs=0)")
        
        # ============================================================
        # Stage 2: Main Training - Primary training phase
        # ============================================================
        # Support both new config format (main_epochs) and legacy format (warmup_run_epochs)
        main_epochs = self.config.get('training.main_epochs', None)
        if main_epochs is None:
            # Legacy support: warmup_run_epochs was the total for "warmup" phase
            # We've already done warmup_epochs, so subtract it
            legacy_warmup_run = self.config.get('training.warmup_run_epochs', None)
            if legacy_warmup_run is not None:
                main_epochs = max(0, legacy_warmup_run - warmup_epochs)
                logger.info(f"Using legacy config: main_epochs={main_epochs} (warmup_run_epochs - warmup_epochs)")
            else:
                main_epochs = 60  # Default
        
        if main_epochs > 0:
            with log_stage(f"Main Training Stage ({main_epochs} epochs)"):
                trainer.train(num_epochs=main_epochs, verbose=True)
            
            # Cross-correlation analysis after main training
            try:
                xcorr_res = compute_xcorr_best_lag(trainer, nt="fmri", node_idx=0, feat_idx=0)
                logger.info(f"Cross-correlation result: {xcorr_res}")
            except Exception as e:
                logger.warning(f"Cross-correlation analysis failed: {e}")
        else:
            logger.info("Main training stage skipped (main_epochs=0)")
        
        # ============================================================
        # Stage 3: Fine-tuning - Refine with stronger temporal weight
        # ============================================================
        finetune_epochs = self.config.get('training.finetune_epochs', 30)
        
        if finetune_epochs > 0:
            # Increase temporal weight for fine-tuning
            finetune_temp_weight = self.config.get('training.finetune_temp_weight', None)
            if finetune_temp_weight is None:
                # Fallback to loss.temp_weight for backward compatibility
                finetune_temp_weight = self.config.get('loss.temp_weight', 5.0)
            
            original_temp_weight = trainer.temp_weight
            trainer.temp_weight = finetune_temp_weight
            
            logger.info(f"Fine-tuning: Increased temp_weight from {original_temp_weight} to {finetune_temp_weight}")
            
            with log_stage(f"Fine-tuning Stage ({finetune_epochs} epochs, temp_weight={finetune_temp_weight})"):
                trainer.train(num_epochs=finetune_epochs, verbose=True)
        else:
            logger.info("Fine-tuning stage skipped (finetune_epochs=0)")
        
        # ============================================================
        # Save final model
        # ============================================================
        if self.config.get('output.save_final_model', True):
            model_path = paths["hetero_model"]
            try:
                trainer.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
    
    def run(self):
        """Run training workflow for all subjects."""
        # Load atlas
        atlas_path = self.base_dir.parent / "schaefer200_mask_ready.json"
        atlas = load_atlas(atlas_path)
        logger.info(f"Loaded atlas from {atlas_path}")
        
        # Process each subject
        for subject_dir in self.subjects:
            try:
                self.train_subject(subject_dir, atlas)
            except Exception as e:
                logger.error(f"Failed to process {subject_dir.name}: {e}", exc_info=True)
                continue
        
        logger.info("Training workflow completed for all subjects")


def run_training(config: Config, base_dir: Optional[Path] = None):
    """
    Main entry point for training workflow.
    
    Args:
        config: Configuration object
        base_dir: Base directory (defaults to config or test_file3)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "test_file3"
    
    workflow = TrainingWorkflow(config, base_dir)
    workflow.run()
