import torch
import numpy as np
import torch.nn.functional as F

from lib.training.hyperdict import HDict
from lib.models.pcqm.gap_predictor import TGT_Gap
from lib.data.pcqm import data
from lib.data.pcqm.structural_transform import AddStructuralData
from ..commons import BinsProcessor
from ..tgt_training import TGTTraining

class SCHEME(TGTTraining):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            save_path_prefix    = HDict.L(lambda c: 'models/pcqm/gap_pred' if c.model_prefix is None else f'models/pcqm/{c.model_prefix}/gap_pred'),
            embed_3d_type       = 'gaussian',
            train_split         = 'train',
            val_split           = 'valid',
            test_split          = 'test-dev',
            bins_input_path     = None,
            bins_shift_half     = True,
            bins_zero_diag      = True,
        )
        return config_dict
    
    def __post_init__(self):
        super().__post_init__()
        if self.executing_command == 'evaluate':
            self.nb_draw_samples = self.config.prediction_samples
        if self.config.bins_input_path is not None:
            self.bins_proc = BinsProcessor(self.config.bins_input_path,
                                           shift_half=self.config.bins_shift_half,
                                           zero_diag=self.config.bins_zero_diag)
        
    
    def get_dataset_config(self, split):
        dataset_config, _ = super().get_dataset_config(split)
        ds_split = {'train':self.config.train_split,
                    'val':self.config.val_split,
                    'test':self.config.test_split}[split]
        
        columns = [data.Bins(self.bins_proc.data_path,
                             self.bins_proc.num_samples)]
        
        is_train_split = 'train' in split
        
        transforms = [AddStructuralData()]
        dataset_config.update(
            return_idx          = True,
            split               = ds_split,
            dataset_path        = self.config.dataset_path,
            transforms          = transforms,
            verbose             = int(self.is_main_rank and is_train_split),
            additional_columns  = columns,
        )
        return dataset_config, data.PCQM4Mv2Dataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, TGT_Gap
    
    def preprocess_batch(self, batch, training):
        batch = super().preprocess_batch(batch, training)
        
        node_mask = batch['node_mask']
        edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        batch['edge_mask'] = edge_mask

        bins = batch['dist_bins']
        
        dist_input = self.bins_proc.bins2dist(bins)
        batch['dist_input'] = dist_input
        
        return batch
    
    
    def prediction_step(self, batch):
        gap_pred = []
        nb_samples = self.nb_draw_samples
        valid_samples = 0
        
        all_dist_inputs = batch['dist_input']
        assert all_dist_inputs.ndim == 4
        num_dist_inputs = all_dist_inputs.size(1)
        for _ in range(nb_samples*2):
            batch['dist_input'] = all_dist_inputs[:,valid_samples%num_dist_inputs]
            new_gap_pred = self.model(batch)
            if torch.isnan(new_gap_pred).any() or torch.isinf(new_gap_pred).any():
                continue
            
            gap_pred.append(new_gap_pred)
            
            valid_samples += 1
            if valid_samples >= nb_samples:
                break
        
        if not valid_samples:
            raise ValueError('All predictions were NaN')
        elif valid_samples < nb_samples:
            nan_samples = nb_samples - valid_samples
            print(f'Warning: '
                  f'{nan_samples}/{nb_samples} predictions were NaN')
        
        gap_pred = torch.stack(gap_pred, dim=-1)
        return dict(
            idx = batch['idx'],
            gap_pred = gap_pred,
            gap_target = batch['target'],
        )
    
    def prediction_loop(self, dataloader):
        return super().prediction_loop(dataloader,
                       predict_in_train=self.config.predict_in_train)
    
    def evaluate_predictions(self, predictions,
                             dataset_name='validation',
                             evaluation_stage=False):
        gap_pred = np.mean(predictions['gap_pred'], axis=-1)
        gap_target = predictions['gap_target']
        if dataset_name == 'test':
            from ogb.lsc.pcqm4mv2 import PCQM4Mv2Evaluator
            evaluator = PCQM4Mv2Evaluator()
            evaluator.save_test_submission(
                input_dict = {'y_pred': gap_pred},
                dir_path = self.config.predictions_path,
                mode = 'test-dev',
            )
            print(f'Saved final test-dev predictions to {self.config.predictions_path}')
            return {'loss': np.nan}
        
        loss = np.abs(gap_pred - gap_target).mean()
        return dict(
            loss = loss,
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if self.is_main_rank: print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions,
                                            dataset_name=dataset_name,
                                            evaluation_stage=True)
        return results    
    
    def execute_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_checkpoint()
        self.model_summary()
        if self.is_main_rank:
            self.save_checkpoint(save_state=False,
                                 save_model=True,
                                 save_optimizer=False,
                                 save_grad_scaler=False,
                                 save_backup_checkpoint=False)
    
    def make_predictions(self):
        super().make_predictions()
        self.evaluate_and_save()
