import torch
import numpy as np
import torch.nn.functional as F

from lib.training.hyperdict import HDict
from lib.models.pcqm.multitask import TGT_Multi
from lib.data.pcqm import data
from lib.data.pcqm.structural_transform import AddStructuralData
from ..tgt_training import TGTTraining
from ..commons import DiscreteDistLoss, add_coords_noise, coords2dist

class SCHEME(TGTTraining):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            save_path_prefix    = 'models/pcqm_pretrain',
            dataset_path        = 'data/PCQM',
            prediction_samples  = 1,
            predict_in_train    = HDict.L(lambda c: c.prediction_samples > 1),
            coords_noise        = 0.5,
            coords_noise_smooth = 1.0,
            embed_3d_type       = 'gaussian',
            num_dist_bins       = 256,
            range_dist_bins     = 8,
            train_split         = 'train-3d',
            val_split           = 'valid-3d',
            test_split          = 'test-dev',
            dist_loss_weight    = 0.1,
        )
        return config_dict
    
    def __post_init__(self):
        super().__post_init__()
        self.dist_loss_fn = DiscreteDistLoss(num_bins=self.config.num_dist_bins,
                                             range_bins=self.config.range_dist_bins)
    
    def get_dataset_config(self, split):
        dataset_config, _ = super().get_dataset_config(split)
        ds_split = {'train':self.config.train_split,
                    'val':self.config.val_split,
                    'test':self.config.test_split}[split]
        
        columns = [data.Coords('dft')]
        
        is_train_split = 'train' in split
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

        coords = batch['dft_coords']
        coords = add_coords_noise(coords=coords,
                                  edge_mask=edge_mask,
                                  noise_level=self.config.coords_noise,
                                  noise_smoothing=self.config.coords_noise_smooth)
        
        dist_input = coords2dist(coords)
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
        dist_probs = None
        nb_samples = self.config.prediction_samples
        valid_samples = 0
        for _ in range(nb_samples*2):
            new_gap_pred, new_dist_logits = self.model(batch)
            if torch.isnan(new_gap_pred).any() or torch.isinf(new_gap_pred).any()\
                or torch.isnan(new_dist_logits).any() or torch.isinf(new_dist_logits).any():
                continue
            
            if gap_pred is None:
                gap_pred = new_gap_pred
            else:
                gap_pred.add_(new_gap_pred)
            
            new_dist_probs = F.softmax(new_dist_logits, dim=-1)
            if dist_probs is None:
                dist_probs = new_dist_probs
            else:
                dist_probs.add_(new_dist_probs)
            
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
        
        dist_probs = dist_probs + dist_probs.transpose(-2,-3)
        dist_probs.div_(valid_samples*2)
        dist_targ = coords2dist(batch['dft_coords'])
        dist_logits = torch.log(dist_probs + 1e-9)
        dist_loss = self.dist_loss_fn(dist_logits, dist_targ, batch['edge_mask'],
                                      reduce=False)
        return dict(
            gap_loss = gap_loss,
            dist_loss = dist_loss,
        )
    
    def prediction_loop(self, dataloader):
        return super().prediction_loop(dataloader,
                       predict_in_train=self.config.predict_in_train)
    
    def evaluate_predictions(self, predictions,
                             dataset_name='validation',
                             evaluation_stage=False):
        gap_loss = predictions['gap_loss'].mean()
        dist_loss = predictions['dist_loss'].mean()
        loss = gap_loss \
                + self.config.dist_loss_weight * dist_loss
        return dict(
            gap_loss = gap_loss,
            dist_loss = dist_loss,
            loss = loss,
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if self.is_main_rank: print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions,
                                            dataset_name=dataset_name,
                                            evaluation_stage=True)
        return results    

