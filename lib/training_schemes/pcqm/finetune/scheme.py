import torch
import numpy as np
import torch.nn.functional as F

from lib.training.hyperdict import HDict
from lib.models.pcqm.multitask import TGT_Multi
from lib.data.pcqm import data
from lib.data.pcqm.structural_transform import AddStructuralData
from ..commons import DiscreteDistLoss, coords2dist, BinsProcessor
from ..tgt_training import TGTTraining

class SCHEME(TGTTraining):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            save_path_prefix    = HDict.L(lambda c: 'models/pcqm/finetune' if c.model_prefix is None else f'models/pcqm/{c.model_prefix}/finetune'),
            embed_3d_type       = 'gaussian',
            num_dist_bins       = 256,
            range_dist_bins     = 8,
            train_split         = 'train',
            val_split           = 'valid',
            test_split          = 'test-dev',
            dist_loss_weight    = 0.1,
            bins_input_path     = None,
            bins_shift_half     = True,
            bins_zero_diag      = True,
        )
        return config_dict
    
    def __post_init__(self):
        super().__post_init__()
        if self.executing_command == 'evaluate':
            self.nb_draw_samples = self.config.prediction_samples
        self.dist_loss_fn = DiscreteDistLoss(num_bins=self.config.num_dist_bins,
                                             range_bins=self.config.range_dist_bins)
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
        if is_train_split and self.executing_command == 'train':
            columns.append(data.Coords('dft'))
        
        transforms = [AddStructuralData()]
        dataset_config.update(
            split               = ds_split,
            dataset_path        = self.config.dataset_path,
            transforms          = transforms,
            verbose             = int(self.is_main_rank and is_train_split),
            additional_columns  = columns,
        )
        return dataset_config, data.PCQM4Mv2Dataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        model_config.update(
            num_dist_bins = self.config.num_dist_bins,
        )
        return model_config, TGT_Multi
    
    def preprocess_batch(self, batch, training):
        batch = super().preprocess_batch(batch, training)
        
        node_mask = batch['node_mask']
        edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        batch['edge_mask'] = edge_mask

        if training:
            all_bins = batch['dist_bins']
            cur_sample = self.state.current_epoch % all_bins.size(1)
            bins = all_bins[:,cur_sample]
        else:
            bins = batch['dist_bins']
        
        dist_input = self.bins_proc.bins2dist(bins)
        batch['dist_input'] = dist_input
        
        return batch
    
    def calculate_loss(self, outputs, inputs):
        gap_pred, dist_logits = outputs
        
        prim_loss = F.l1_loss(gap_pred, inputs['target'])
        
        dist_targ = coords2dist(inputs['dft_coords'])
        dist_loss = self.dist_loss_fn(dist_logits, dist_targ, inputs['edge_mask'])
        
        loss = prim_loss \
               + self.config.dist_loss_weight * dist_loss
        return loss
    
    
    def prediction_step(self, batch):
        gap_pred = None
        nb_samples = self.nb_draw_samples
        valid_samples = 0
        
        all_dist_inputs = batch['dist_input']
        assert all_dist_inputs.ndim == 4
        num_dist_inputs = all_dist_inputs.size(1)
        for _ in range(nb_samples*2):
            batch['dist_input'] = all_dist_inputs[:,valid_samples%num_dist_inputs]
            new_gap_pred, _ = self.model(batch)
            if torch.isnan(new_gap_pred).any() or torch.isinf(new_gap_pred).any():
                continue
            
            if gap_pred is None:
                gap_pred = new_gap_pred
            else:
                gap_pred.add_(new_gap_pred)
            
            valid_samples += 1
            if valid_samples >= nb_samples:
                break
        
        if not valid_samples:
            raise ValueError('All predictions were NaN')
        elif valid_samples < nb_samples:
            nan_samples = nb_samples - valid_samples
            print(f'Warning: '
                  f'{nan_samples}/{nb_samples} predictions were NaN')
        
        gap_pred.div_(valid_samples)
        gap_loss = torch.abs(gap_pred - batch['target'])
        return dict(
            gap_loss = gap_loss,
        )
    
    def prediction_loop(self, dataloader):
        return super().prediction_loop(dataloader,
                       predict_in_train=self.config.predict_in_train)
    
    def evaluate_predictions(self, predictions,
                             dataset_name='validation',
                             evaluation_stage=False):
        loss = predictions['gap_loss'].mean()
        return dict(
            loss = loss,
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if self.is_main_rank: print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions,
                                            dataset_name=dataset_name,
                                            evaluation_stage=True)
        return results    
    
    def make_predictions(self):
        super().make_predictions()
        self.evaluate_and_save()

