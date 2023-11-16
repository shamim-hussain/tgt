import torch
import numpy as np
from torch.utils.data import DataLoader
import os.path as osp
from tqdm import tqdm

from .hyperdict import HDict
from .training import TrainingBase, DistributedTestDataSampler, cached_property
from . import config as cfg

class TestingBase(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            state_file         = None,
            predict_on         = ['val', 'test'],
            evaluate_on        = HDict.L(lambda c: c.predict_on),
            predictions_path   = HDict.L(lambda c: osp.join(c.save_path,"predictions")),
        )
        return config
    
    @cached_property
    def test_dataset(self):
        return self.get_dataset('test')
    
    @cached_property
    def train_pred_dataloader(self):
        dataset = self.train_dataset_subset
        prediction_batch_size = round(self.config.batch_size*self.config.prediction_bmult)
        common_kwargs = dict(
            dataset=dataset,
            collate_fn=self.collate_fn \
                        if not isinstance(self.collate_fn, dict) \
                            else self.collate_fn['train'],
            pin_memory=True,
        )
        
        # Multiprocess dataloader logic
        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
                multiprocessing_context=self.config.dataloader_mp_context,
            )
        
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs,
                                    batch_size=prediction_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    )
        else:
            sampler = DistributedTestDataSampler(data_source=dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(**common_kwargs, batch_sampler=sampler)
        return dataloader
    
    @cached_property
    def test_dataloader(self):
        prediction_batch_size = round(self.config.batch_size*self.config.prediction_bmult)
        common_kwargs = dict(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn \
                        if not isinstance(self.collate_fn, dict) \
                            else self.collate_fn['test'],
            pin_memory=True,
        )
        
        # Multiprocess dataloader logic
        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
                multiprocessing_context=self.config.dataloader_mp_context,
            )
        
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs,
                                    batch_size=prediction_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    )
        else:
            sampler = DistributedTestDataSampler(data_source=self.test_dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(**common_kwargs, batch_sampler=sampler)
        return dataloader
    
    
    def test_dataloader_for_dataset(self, dataset_name):
        if dataset_name == 'train':
            return self.train_pred_dataloader
        elif dataset_name == 'val':
            return self.val_dataloader
        elif dataset_name == 'test':
            return self.test_dataloader
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')
    
    def predict_and_save(self):
        for dataset_name in self.config.predict_on:
            if self.is_main_rank:
                print(f'Predicting on {dataset_name} dataset...')
            dataloader = self.test_dataloader_for_dataset(dataset_name)
            outputs = self.prediction_loop(dataloader)
            outputs = self.preprocess_predictions(outputs)
        
            if self.is_distributed:
                outputs = self.distributed_gather_predictions(outputs)
            
            if self.is_main_rank:
                predictions = self.postprocess_predictions(outputs)
                self.save_predictions(dataset_name, predictions)
    
    
    def load_model_state(self):
        if self.config.state_file is None:
            state_file = osp.join(self.config.checkpoint_path, 'model_state')
        else:
            state_file = self.config.state_file
        self.base_model.load_state_dict(torch.load(state_file))
        
        if self.is_main_rank:
            print(f'Loaded model state from {state_file}')
        
    def prepare_for_testing(self):
        self.config_summary()
        self.load_model_state()
        
    def make_predictions(self):
        self.prepare_for_testing()
        self.predict_and_save()
    
    
    def get_dataset(self, dataset_name):
        if dataset_name == 'train':
            return self.train_dataset
        elif dataset_name == 'val':
            return self.val_dataset
        elif dataset_name == 'test':
            return self.test_dataset
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        raise NotImplementedError
    
    def evaluate_and_save(self):
        if not self.is_main_rank:
            return
        
        results = {}
        results_file = osp.join(self.config.predictions_path, 'results.yaml')
        
        for dataset_name in self.config.evaluate_on:
            dataset = self.get_dataset(dataset_name)
            predictions = torch.load(osp.join(self.config.predictions_path, f'{dataset_name}.pt'))
            dataset_results = self.evaluate_on(dataset_name, dataset, predictions)

            for k, v in dataset_results.items():
                if isinstance(v, np.ndarray) or isinstance(v, np.number):
                    dataset_results[k] = v.tolist()
                    
            for k,v in dataset_results.items():
                print(f'{dataset_name} {k}: {v}')
            
            results[dataset_name] = dataset_results
            cfg.safe_dump(results, results_file)
    
    
    def do_evaluations(self):
        self.make_predictions()
        self.evaluate_and_save()
    
