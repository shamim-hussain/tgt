from lib.training.training import TrainingBase, cached_property
from lib.training.samplers import DistributedTrainDataSampler as TrainSampler
from lib.training.testing import TestingBase
from lib.training.training_mixins import MonitorBest, LinearLRWarmupCosineDecay
from lib.training.hyperdict import HDict
import os.path as osp
import torch
from lib.data.dataset import padded_collate


class TGTTraining(LinearLRWarmupCosineDecay,MonitorBest,TestingBase,TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name              = 'tgt'        ,
            model_height            = 4            ,
            node_width              = 64           ,
            edge_width              = 8            ,
            num_heads               = 8            ,
            node_act_dropout        = 0.           ,
            edge_act_dropout        = 0.           ,
            source_dropout          = 0.           ,
            drop_path               = 0.           ,
            activation              = 'gelu'       ,
            scale_degree            = True         ,
            node_ffn_multiplier     = 1.           ,
            edge_ffn_multiplier     = 1.           ,
            layer_multiplier        = 1            ,
            upto_hop                = 32           ,
            triplet_heads           = 0            ,
            triplet_type            = 'aggregate'  ,
            triplet_dropout         = 0            ,
            
            embed_3d_type           = 'gaussian'   ,
            num_3d_kernels          = 128          ,
            
            allocate_max_batch      = True         ,
            save_all_checkpoints    = True         ,
            predict_on              = ['val']      ,
            num_epochs              = 1000         ,
            pretrained_weights_file = None         ,
        )
        return config
    
    def __post_init__(self):
        self._nan_loss_count = 0
    
    def get_dataset_config(self, split):
        if self.is_distributed and split == 'train':
            cache_range_fn = TrainSampler.get_slice4len
        else:
            cache_range_fn = None
        config = dict(
            cache_range_fn = cache_range_fn,
        )
        return config, None
    
    def get_model_config(self):
        config = self.config
        model_config = dict(
            model_height        = config.model_height                ,
            node_width          = config.node_width                  ,
            edge_width          = config.edge_width                  ,
            num_heads           = config.num_heads                   ,
            node_act_dropout    = config.node_act_dropout            ,
            edge_act_dropout    = config.edge_act_dropout            ,
            source_dropout      = config.source_dropout              ,
            drop_path           = config.drop_path                   ,
            activation          = config.activation                  ,
            scale_degree        = config.scale_degree                ,
            node_ffn_multiplier = config.node_ffn_multiplier         ,
            edge_ffn_multiplier = config.edge_ffn_multiplier         ,
            layer_multiplier    = config.layer_multiplier            ,
            upto_hop            = config.upto_hop                    ,
            triplet_heads       = config.triplet_heads               ,
            triplet_type        = config.triplet_type                ,
            triplet_dropout     = config.triplet_dropout             ,
            
            num_3d_kernels      = config.num_3d_kernels              ,
            embed_3d_type       = config.embed_3d_type               ,
        )
        return model_config, None
    
    @cached_property
    def train_sampler(self):
        dataset = self.train_dataset_subset
        return TrainSampler(dataset, shuffle=True)
    
    
    def get_dataset(self, split):
        dataset_config, dataset_class = self.get_dataset_config(split)
        if dataset_class is None:
            raise NotImplementedError('Dataset class not specified')
        dataset = dataset_class(**dataset_config)
        return dataset
    
    @property
    def collate_fn(self):
        return padded_collate
    
    def get_base_model(self):
        model_config, model_class = self.get_model_config()
        if model_class is None:
            raise NotImplementedError
        model = model_class(**model_config)
        return model
    
    def prepare_for_training(self):
        super().prepare_for_training()
        
        # GPU memory cache for biggest batch
        if self.config.allocate_max_batch:
            if self.is_main_rank: print('Allocating cache for max batch size...', flush=True)
            torch.cuda.empty_cache()
            self.model.train()
            max_batch = self.collate_fn([self.train_dataset[self.train_dataset.max_nodes_index]]
                                                        * self.config.batch_size)
            max_batch = self.preprocess_batch(batch=max_batch, training=True)
            
            outputs = self.model(max_batch)
            loss = self.calculate_loss(outputs=outputs, inputs=max_batch)
            loss.backward()
            
            for param in self.model.parameters():
                param.grad = None
    
    def initialize_losses(self, logs, training):
        self._total_loss = 0.
        self._total_samples = 0.
    
    def update_losses(self, i, loss, inputs, logs, training):
        step_samples = float(inputs['num_nodes'].shape[0])
        if not self.is_distributed:
            step_loss = loss.item() * step_samples
        else:
            step_samples = torch.tensor(step_samples, device=loss.device,
                                        dtype=loss.dtype)
            
            if training:
                loss = loss.detach()
            step_loss = loss * step_samples
            
            torch.distributed.all_reduce(step_loss)
            torch.distributed.all_reduce(step_samples)
            
            step_loss = step_loss.item()
            step_samples = step_samples.item()
        
        if self.config.mixed_precision:
            if step_loss == step_loss or self._nan_loss_count >= 10:
                self._nan_loss_count = 0
                self._total_loss += step_loss
                self._total_samples += step_samples
            else:
                self._nan_loss_count += 1
        else:
            self._total_loss += step_loss
            self._total_samples += step_samples
        
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/(self._total_samples+1e-12))
    
    
    def load_checkpoint(self, edit_weights_fn=None):
        super().load_checkpoint(load_pretrained_weights=False)
        w_file = self.config.pretrained_weights_file
        if w_file is not None and self.state.global_step == 0:
            weights = torch.load(w_file)
            if edit_weights_fn is not None:
                weights = edit_weights_fn(weights)
                        
            missing, unexpected = self.base_model.load_state_dict(weights, strict=False)
            torch.cuda.empty_cache()
            if self.is_main_rank:
                print(f'Loaded pretrained weights from {w_file}',flush=True)
                print(f'missing keys: {missing}',flush=True)
                print(f'unexpected keys: {unexpected}',flush=True)

