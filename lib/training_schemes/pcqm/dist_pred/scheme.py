import torch
import numpy as np
import torch.nn.functional as F

from lib.training.hyperdict import HDict
from lib.models.pcqm.distance_predictor import TGT_Distance
from lib.data.pcqm import data
from lib.data.pcqm.structural_transform import AddStructuralData
from lib.data.pcqm import bin_ops as pbins
from ..tgt_training import TGTTraining
from ..commons import DiscreteDistLoss, add_coords_noise, coords2dist

class SCHEME(TGTTraining):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            save_path_prefix    = 'models/pcqm_dist_pred',
            dataset_path        = 'data/PCQM',
            prediction_samples  = 10,
            predict_in_train    = True,
            predict_on          = ['train', 'val'],
            coords_noise        = 0,
            coords_noise_smooth = 0,
            coords_input        = 'rdkit',
            coords_target       = 'dft',
            embed_3d_type       = HDict.L(lambda c: 'gaussian' if c.coords_input!='none'
                                                     else 'none'),
            num_dist_bins       = 512,
            range_dist_bins     = 8,
            train_split         = 'train-3d' if self.executing_command == 'train' else 'train',
            val_split           = 'valid-3d' if self.executing_command == 'train' else 'valid',
            test_split          = 'test-dev',
            save_pred_dir       = HDict.L(lambda c: f'bins{c.prediction_samples}'),
            save_bins_smooth    = 0,
            coords_target_noise = 0,
        )
        return config_dict
    
    def __post_init__(self):
        super().__post_init__()
        self.xent_loss_fn = DiscreteDistLoss(num_bins=self.config.num_dist_bins,
                                             range_bins=self.config.range_dist_bins)
        
        self._uses_3d = self.config.coords_input != 'none'
        
        executing_training = self.executing_command == 'train'
        
        input_dist_rdkit = self.config.coords_input == 'rdkit'
        predict_dist_rdkit = self.config.coords_target == 'rdkit'
        self._use_rdkit_coords = input_dist_rdkit or (predict_dist_rdkit and executing_training)
        
        input_dist_dft = self.config.coords_input == 'dft'
        predict_dist_dft = self.config.coords_target == 'dft'
        self._use_dft_coords = input_dist_dft or (predict_dist_dft and executing_training)
    
    def get_dataset_config(self, split):
        dataset_config, _ = super().get_dataset_config(split)
        ds_split = {'train':self.config.train_split,
                    'val':self.config.val_split,
                    'test':self.config.test_split}[split]
        
        columns = []
        if self._use_rdkit_coords:
            columns.append(data.Coords('rdkit'))
        
        if self._use_dft_coords:
            columns.append(data.Coords('dft'))
        
        is_train_split = 'train' in split
        transforms = [AddStructuralData()]
        dataset_config.update(
            split               = ds_split,
            dataset_path        = self.config.dataset_path,
            return_idx          = True,
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
        return model_config, TGT_Distance
    
    def preprocess_batch(self, batch, training):
        batch = super().preprocess_batch(batch, training)
        
        node_mask = batch['node_mask']
        edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        batch['edge_mask'] = edge_mask
        if self._uses_3d:
            coords = self.get_coords_input(batch, training)
            if self.config.coords_noise > 0:
                coords = add_coords_noise(coords=coords,
                                          edge_mask=edge_mask,
                                          noise_level=self.config.coords_noise,
                                          noise_smoothing=self.config.coords_noise_smooth)
            
            dist_input = coords2dist(coords)
            batch['dist_input'] = dist_input
        
        return batch


    def get_coords_input(self, batch, training):
        if self.config.coords_input == 'rdkit':
            coords = batch['rdkit_coords']
        elif self.config.coords_input == 'dft':
            coords = batch['dft_coords']
        else:
            raise ValueError('Invalid coords_input')
        return coords
    
    def get_coords_target(self, batch):
        if self.config.coords_target == 'rdkit':
            coords = batch['rdkit_coords']
        elif self.config.coords_target == 'dft':
            coords = batch['dft_coords']
        else:
            raise ValueError('Invalid coords_target')
        return coords
    
    def get_dist_target(self, batch, training=False):
        coords_targ = self.get_coords_target(batch)
        noise_level = self.config.coords_target_noise
        if training and noise_level > 0:
            noise = coords_targ.new(coords_targ.size())\
                               .normal_(0, noise_level)
            coords_targ = coords_targ + noise
        dist_targ = coords2dist(coords_targ)
        return dist_targ
    
    def calculate_loss(self, outputs, inputs):
        dist_targ = self.get_dist_target(inputs, training=True)
        mask = inputs['edge_mask']
        loss = self.xent_loss_fn(outputs, dist_targ, mask)
        return loss
    
    def predict_probs(self, batch):
        probs = None
        nb_samples = self.config.prediction_samples
        valid_samples = 0
        for _ in range(nb_samples*2):
            new_logits = self.model(batch)
            if torch.isnan(new_logits).any() or torch.isinf(new_logits).any():
                continue
            
            new_probs = F.softmax(new_logits, dim=-1)
            if probs is None:
                probs = new_probs
            else:
                probs.add_(new_probs)
            
            valid_samples += 1
            if valid_samples >= nb_samples:
                break
        
        if not valid_samples:
            raise ValueError('All predictions were NaN')
        elif valid_samples < nb_samples:
            nan_samples = nb_samples - valid_samples
            print(f'Warning: '
                  f'{nan_samples}/{nb_samples} predictions were NaN')
        
        probs = probs + probs.transpose(-2,-3)
        probs.div_(valid_samples*2)
        return probs
    
    
    def prediction_step4eval(self, batch):
        probs = self.predict_probs(batch)
        dist_targ = self.get_dist_target(batch)
        logits = torch.log(probs + 1e-9)
        mask = batch['edge_mask']
        xent = self.xent_loss_fn(logits, dist_targ, mask, reduce=False)
        return dict(
            loss = xent,
        )
    
    
    def predict_bins(self, batch):
        bins = []
        nb_samples = self.config.prediction_samples
        valid_samples = 0
        for _ in range(nb_samples*2):
            logits = self.model(batch)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue
            
            new_prob = torch.softmax(logits, dim=-1)
            new_prob = new_prob + new_prob.transpose(-2,-3)
            
            new_bins = new_prob.argmax(dim=-1)
            bins.append(new_bins)
            
            valid_samples += 1
            if valid_samples >= nb_samples:
                break
        
        if valid_samples < nb_samples:
            nan_samples = nb_samples - valid_samples
            raise ValueError(f'{nan_samples}/{nb_samples} predictions were NaN')
        
        bins = torch.stack(bins, dim=1)
        return bins
        
    
    def prediction_step4savebins(self, batch):
        bins = self.predict_bins(batch)
        
        idx = batch['idx'].cpu().numpy()
        num_nodes = batch['num_nodes'].cpu().numpy()
        bins = bins.cpu().numpy()
        
        if self.config.num_dist_bins <= 256:
            bins = bins.astype('uint8')
        elif self.config.num_dist_bins <= 65536:
            bins = bins.astype('uint16')
        
        packed_bins = []
        for i,n in enumerate(num_nodes):
            packed_bins_i = pbins.pack_bins_multi(bins[i,:,:n,:n])
            packed_bins_i = packed_bins_i.reshape(-1)
            packed_bins.append(packed_bins_i)
        
        return dict(
            idx = idx,
            bins = packed_bins,
        )
        
    def prediction_step(self, batch):
        if self.executing_command == 'predict':
            return self.prediction_step4savebins(batch)
        else:
            return self.prediction_step4eval(batch)
    
    def prediction_loop(self, dataloader):
        return super().prediction_loop(dataloader,
                       predict_in_train=self.config.predict_in_train)
    
    def evaluate_predictions(self, predictions,
                             dataset_name='validation',
                             evaluation_stage=False):
        loss = predictions['loss'].mean()
        return dict(
            loss = loss,
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if self.is_main_rank: print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions,
                                            dataset_name=dataset_name,
                                            evaluation_stage=True)
        return results    
    
    def predict_and_save(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import os
        
        def collate_columns(predictions, column_names):
            collated = {col:[] for col in column_names}
            for batch in predictions:
                for col in column_names:
                    collated[col].extend(batch[col])
            return collated
        
        save_pred_dir = os.path.join(self.config.predictions_path,
                                     self.config.save_pred_dir)
        os.makedirs(save_pred_dir, exist_ok=True)
        data_dir = os.path.join(save_pred_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        meta_path = os.path.join(save_pred_dir, 'meta.json')
        for dataset_name in self.config.predict_on:
            if self.is_main_rank:
                print(f'Predicting on {dataset_name} dataset...')
            dataloader = self.test_dataloader_for_dataset(dataset_name)
            outputs = self.prediction_loop(dataloader)
            
            table_dict = dict(
                idx = np.concatenate([o['idx'] for o in outputs]),
            )
                
            columns = collate_columns(outputs, ['bins'])
            table_dict.update(columns)

            if self.is_main_rank:
                import json
                meta = dict(
                    num_bins = self.config.num_dist_bins,
                    range_bins = self.config.range_dist_bins,
                    num_samples = self.config.prediction_samples,
                )
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
            
            
            table = pa.Table.from_pydict(table_dict)
            save_path = os.path.join(data_dir, 
                                     f'{dataset_name}_{self.ddp_rank:03d}.parquet')
            pq.write_table(table, save_path)
            print(f'Rank {self.ddp_rank} saved {dataset_name} to {save_path}')
            

