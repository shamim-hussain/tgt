import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Sampler
from contextlib import nullcontext

import os
import os.path as osp
from datetime import datetime

from .hyperdict import HDict
from . import config as cfg
from .samplers import DistributedTestDataSampler
from .utils import cached_property, state_dict_to_cpu

class StopTrainingException(Exception):
    pass


class TrainingBase:
    def __init__(self, config=None, ddp_rank=0, ddp_world_size=1, command=None):
        self.config_input = config
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.executing_command = command
        
        self.is_distributed = (self.ddp_world_size > 1)
        self.is_main_rank = (self.ddp_rank == 0)
        self.recovery_tries = 0
        
        
        self.config_dict = self.get_default_config()
        self.config_dict.inherit_from(config)
        self.config = self.config_dict.to_namespace()
        
        self.state = self.get_default_state()
        
        self.__post_init__()
    
    def __post_init__(self):
        pass

    def get_dataset(self, split):
        raise NotImplementedError
    
    @cached_property
    def train_dataset(self):
        return self.get_dataset('train')
    
    @cached_property
    def val_dataset(self):
        return self.get_dataset('val')
    
    @cached_property
    def train_dataset_subset(self):
        dataset = self.train_dataset
        if self.config.trial_run.perform_trial_run:
            num_samples = self.config.trial_run.num_train_samples * self.ddp_world_size
            dataset = Subset(dataset, list(range(min(len(dataset), num_samples))))
        return dataset
    
    @cached_property
    def val_dataset_subset(self):
        dataset = self.val_dataset
        if self.config.trial_run.perform_trial_run:
            num_samples = self.config.trial_run.num_val_samples * self.ddp_world_size
            dataset = Subset(dataset, list(range(min(len(dataset), num_samples))))
        return dataset
    
    @cached_property
    def collate_fn(self):
        return None

    @cached_property
    def train_sampler(self):
        dataset = self.train_dataset_subset
        return  torch.utils.data.DistributedSampler(dataset, shuffle=True)
    
    @cached_property
    def train_dataloader(self):
        dataset = self.train_dataset_subset
        common_kwargs = dict(
            dataset=dataset,
            batch_size=self.config.batch_size,
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
            dataloader = DataLoader(**common_kwargs, shuffle=True,
                                    drop_last=False)
        else:
            dataloader = DataLoader(**common_kwargs, 
                                    sampler=self.train_sampler)
        return dataloader
    
    @cached_property
    def val_dataloader(self):
        dataset = self.val_dataset_subset
        common_kwargs = dict(
            dataset=dataset,
            collate_fn=self.collate_fn \
                          if not isinstance(self.collate_fn, dict) \
                              else self.collate_fn['val'],
            pin_memory=True,
        )
        
        # Multiprocess dataloader logic
        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
                multiprocessing_context=self.config.dataloader_mp_context,
            )
        
        prediction_batch_size = round(self.config.batch_size*self.config.prediction_bmult)
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs, 
                                    batch_size=prediction_batch_size,
                                    shuffle=False, drop_last=False)
        else:
            sampler = DistributedTestDataSampler(data_source=dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(**common_kwargs, batch_sampler=sampler)
        return dataloader

    def get_base_model(self):
        raise NotImplementedError
    
    @cached_property
    def base_model(self):
        return self.get_base_model().cuda()
    
    @cached_property
    def model(self):
        model = self.base_model
        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
        return model
    
    @cached_property
    def trainable_params(self):
        return list(p for p in self.model.parameters() if p.requires_grad)
    
    @cached_property
    def optimizer(self):
        config = self.config
        if config.optimizer.lower().startswith('apex_'):
            opt = config.optimizer[5:]
            import apex.optimizers
            optimizer_class = getattr(apex.optimizers, opt)
        else:
            optimizer_class = getattr(torch.optim, config.optimizer)
        optimizer = optimizer_class(self.trainable_params,
                                    lr=config.max_lr,
                                    **config.optimizer_params)
        return optimizer
    
    
    def progbar(self, iterable, total, desc, **kwargs):
        progb_config = self.config.progbar
        miniters = progb_config.miniters
        
        common_kwargs = dict(iterable=iterable,
                             total=total,
                             desc=desc,
                             miniters=miniters,
                             **kwargs)
        
        if progb_config.disable:
            from .progress import Progress
            return Progress(**common_kwargs)
        else:
            from tqdm import tqdm
            common_kwargs.update(dynamic_ncols=True)
            if not progb_config.show_bar:
                common_kwargs.update(bar_format = '{l_bar}{r_bar}')
                
            return tqdm(**common_kwargs)
                        

    def get_default_config(self):
        return HDict(
            scheme                  = None,
            model_name              = 'unnamed_model',
            distributed             = False,
            random_seed             = None,
            num_epochs              = 100,
            save_path_prefix        = 'models',
            save_path               = HDict.L(lambda c: osp.join(c.save_path_prefix,c.model_name)),
            checkpoint_path         = HDict.L(lambda c: osp.join(c.save_path,"checkpoint")),
            config_path             = HDict.L(lambda c: osp.join(c.save_path,"config")),
            summary_path            = HDict.L(lambda c: osp.join(c.save_path,"summary")),
            log_path                = HDict.L(lambda c: osp.join(c.save_path,"logs")),
            validation_frequency    = 1,
            validation_condition    = None,
            batch_size              = HDict.L(lambda c: 128 if c.distributed else 32),
            optimizer               = 'Adam'    ,
            max_lr                  = 5e-4      ,
            clip_grad_value         = None      ,
            clip_grad_norm          = None      ,
            optimizer_params        = {}        ,
            dataloader_workers      = 0         ,
            dataloader_mp_context   = 'forkserver',
            training_type           = 'normal'  ,
            evaluation_type         = 'validation',
            predictions_path        = HDict.L(lambda c: osp.join(c.save_path,"predictions")),
            prediction_bmult        = 1         ,
            mixed_precision         = False     ,
            lr_schedule             = None      ,
            verbose_lr_log          = HDict.L(lambda c: c.lr_schedule is not None),
            pretrained_weights_file = None      ,
            save_all_checkpoints    = False     ,
            all_checkpoints_path    = HDict.L(lambda c: osp.join(c.save_path,"all_checkpoints")),
            max_recovery_tries      = 10        ,
            progbar                 = HDict(
                disable             = False,
                miniters            = 0.05,
                show_bar            = True,
            ),
            trial_run               = HDict(
                perform_trial_run   = False,
                num_train_samples   = HDict.L(lambda c: c.P.batch_size*2),
                num_val_samples     = HDict.L(lambda c: c.P.batch_size*2),
                save_checkpoint     = False,
            )
        )
    
    def get_default_state(self):
        state =  HDict(
            current_epoch = 0,
            global_step = 0,
        )
        return state
    
    def config_summary(self):
        if not self.is_main_rank: return
        for k,v in self.config_dict.to_dict().items():
            print(f'{k} : {v}', flush=True)
    
    def save_config_file(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
        cfg.save_config_to_file(self.config_dict.to_dict(), self.config.config_path+'.yaml')
        cfg.save_config_to_file(self.config_input, self.config.config_path+'_input.yaml')
    
    def model_summary(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.summary_path), exist_ok=True)
        trainable_params = 0
        non_trainable_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()
        summary = dict(
            trainable_params = trainable_params,
            non_trainable_params = non_trainable_params,
            model_representation = repr(self.model),
        )
        cfg.safe_dump(summary, self.config.summary_path+'.txt')
    
    def save_checkpoint(self,
                        save_state=True,
                        save_model=True,
                        save_optimizer=True,
                        save_grad_scaler=True,
                        save_backup_checkpoint=True):
        if not self.is_main_rank:
            return
        if self.config.trial_run.perform_trial_run and not self.config.trial_run.save_checkpoint:
            return
        ckpt_path = self.config.checkpoint_path
        os.makedirs(ckpt_path, exist_ok=True)
        
        if save_state:
            torch.save(self.state, os.path.join(ckpt_path, 'training_state'))
        if save_model:
            model_dict = state_dict_to_cpu(self.base_model.state_dict())
            torch.save(model_dict, os.path.join(ckpt_path, 'model_state'))
        if save_optimizer:
            optim_dict = state_dict_to_cpu(self.optimizer.state_dict())
            torch.save(optim_dict, os.path.join(ckpt_path, 'optimizer_state'))
        if save_grad_scaler and self.grad_scaler is not None:
            torch.save(self.grad_scaler.state_dict(), os.path.join(ckpt_path, 'grad_scaler_state'))
        print(f'Checkpoint saved to: {ckpt_path}',flush=True)
        
        if save_backup_checkpoint and self.config.save_all_checkpoints:
            ckpt_path = os.path.join(self.config.all_checkpoints_path,
                                     f'epoch_{self.state.current_epoch}')
            os.makedirs(ckpt_path, exist_ok=True)
            
            torch.save(self.state, os.path.join(ckpt_path, 'training_state'))
            torch.save(model_dict, os.path.join(ckpt_path, 'model_state'))
            torch.save(optim_dict, os.path.join(ckpt_path, 'optimizer_state'))
            if self.grad_scaler is not None:
                torch.save(self.grad_scaler.state_dict(),
                           os.path.join(ckpt_path, 'grad_scaler_state'))
            print(f'Backup checkpoint saved to: {ckpt_path}',flush=True)
    
    def load_checkpoint(self,
                        strict_weight_check=True,
                        load_optimizer=True,
                        load_model=True,
                        load_state=True,
                        load_grad_scaler=True,
                        load_pretrained_weights=True,
                        ):
        ckpt_path = self.config.checkpoint_path
        if load_state:
            try:
                self.state.update(torch.load(os.path.join(ckpt_path, 'training_state')))
                if self.is_main_rank: print(f'State loaded from: {ckpt_path}',flush=True)
            except FileNotFoundError: pass
        if load_optimizer:
            torch.cuda.empty_cache()
            try:
                self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'optimizer_state')))
                if self.is_main_rank: print(f'Optimizer loaded from: {ckpt_path}',flush=True)
            except FileNotFoundError: pass
            torch.cuda.empty_cache()
        if load_model:
            torch.cuda.empty_cache()
            try:
                self.base_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'model_state')),
                                                strict=strict_weight_check)
                if self.is_main_rank: print(f'Model loaded from: {ckpt_path}',flush=True)
            except FileNotFoundError: pass
            torch.cuda.empty_cache()
        if load_grad_scaler:
            try:
                if self.grad_scaler is not None:
                    self.grad_scaler.load_state_dict(torch.load(os.path.join(ckpt_path, 'grad_scaler_state')))
                    if self.is_main_rank: print(f'Grad scaler loaded from: {ckpt_path}',flush=True)
                    torch.cuda.empty_cache()
            except FileNotFoundError: pass
        if load_pretrained_weights:
            w_file = self.config.pretrained_weights_file
            if w_file is not None and self.state.global_step == 0:
                missing, unexpected = self.base_model.load_state_dict(torch.load(w_file), strict=False)
                torch.cuda.empty_cache()
                if self.is_main_rank:
                    print(f'Loaded pretrained weights from {w_file}',flush=True)
                    print(f'missing keys: {missing}',flush=True)
                    print(f'unexpected keys: {unexpected}',flush=True)
        
    # Callbacks
    def on_train_begin(self):
        pass
    def on_train_end(self):
        pass
    def on_epoch_begin(self, logs, training):
        pass
    def on_epoch_end(self, logs, training):
        pass
    def on_batch_begin(self, i, logs, training):
        pass
    def on_batch_end(self, i, logs, training):
        pass
    
    
    # Logging
    def get_verbose_logs(self):
        return OrderedDict(loss='0.4f')
    
    @cached_property
    def verbose_logs(self):
        return self.get_verbose_logs()
    
    def update_logs(self, logs, training, **updates):
        if training:
            logs.update(updates)
        else:
            logs.update(('val_'+k,v) for k,v in updates.items())
    
    def log_description(self, i, logs, training):
        if training:
            descriptions = list(f'{k} = {logs[k]:{f}}' 
                                for k,f in self.verbose_logs.items())
            if self.config.verbose_lr_log:
                descriptions.append(f'(lr:{logs["lr"]:0.3e})')
        else:
            descriptions = list(f'val_{k} = {logs["val_"+k]:{f}}' 
                                for k,f in self.verbose_logs.items())
        return descriptions
    
    
    # Training loop
    def preprocess_batch(self, batch, training):
        if hasattr(batch, 'cuda'):
            return batch.cuda(non_blocking=True)
        elif hasattr(batch, 'items'):
            return batch.__class__((k,v.cuda(non_blocking=True)) for k,v in batch.items())
        elif hasattr(batch, '__iter__'):
            return batch.__class__(v.cuda(non_blocking=True) for v in batch)
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')
    
    def calculate_loss(self, outputs, inputs):
        raise NotImplementedError
    
    @cached_property
    def train_steps_per_epoch(self):
        return len(self.train_dataloader)
    
    @cached_property
    def grad_scaler(self):
        if self.config.mixed_precision:
            return torch.cuda.amp.GradScaler()
        else:
            return None
    
    @cached_property
    def fwd_pass_context(self):
        return nullcontext if not self.config.mixed_precision\
                           else torch.cuda.amp.autocast
    
    def training_step(self, batch, logs):
        for param in self.trainable_params:
            param.grad = None
        
        mixed_precision = self.config.mixed_precision
        clip_grad_value = self.config.clip_grad_value
        clip_grad_norm = self.config.clip_grad_norm
        
        with self.fwd_pass_context():
            outputs = self.model(batch)
            loss = self.calculate_loss(outputs=outputs, inputs=batch)
        if not mixed_precision:
            loss.backward()
        else:
            self.grad_scaler.scale(loss).backward()
        
        should_clip_grad_value = clip_grad_value is not None
        should_clip_grad_norm = clip_grad_norm is not None
        if mixed_precision and (should_clip_grad_value or should_clip_grad_norm):
            self.grad_scaler.unscale_(self.optimizer)
        
        if should_clip_grad_value:
            nn.utils.clip_grad_value_(self.trainable_params, self.config.clip_grad_value)
        if should_clip_grad_norm:
            nn.utils.clip_grad_norm_(self.trainable_params, self.config.clip_grad_norm)
        
        if not mixed_precision:
            self.optimizer.step()
        else:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        return outputs, loss
    
    def validation_step(self, batch, logs):
        with self.fwd_pass_context():
            outputs = self.model(batch)
            loss = self.calculate_loss(outputs=outputs, inputs=batch)
        return outputs, loss
    
    def initialize_metrics(self, logs, training):
        pass
    
    def update_metrics(self, outputs, inputs, logs, training):
        pass
    
    def initialize_losses(self, logs, training):
        self._total_loss = 0.
    
    def update_losses(self, i, loss, inputs, logs, training):
        if not self.is_distributed:
            step_loss = loss.item()
        else:
            if training:
                loss = loss.detach()
            torch.distributed.all_reduce(loss)
            step_loss = loss.item()/self.ddp_world_size
        self._total_loss += step_loss
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/(i+1))
        
    
    def train_epoch(self, epoch, logs,
                    minimal=False,
                    train_in_eval=False):
        if not train_in_eval:
            self.model.train()
        else:
            self.model.eval()
        
        if not minimal:
            self.initialize_losses(logs, True)
            self.initialize_metrics(logs, True)
        
        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)
        
        gen = self.train_dataloader
        if self.is_main_rank:
            gen = self.progbar(gen, total=self.train_steps_per_epoch,
                               desc='Training: ')
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch=batch, training=True)
                outputs, loss = self.training_step(batch, logs)
                
                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)
                
                if not minimal:
                    self.update_losses(i, loss, batch, logs, True)
                    self.update_metrics(outputs, batch, logs, True)
                
                self.on_batch_end(i, logs, True)
                
                if self.is_main_rank and not minimal:
                    desc = 'Training: '+'; '.join(self.log_description(i, logs, True))
                    gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
            for param in self.trainable_params:
                param.grad = None
    
    
    def validation_epoch(self, epoch, logs):
        self.model.eval()
        self.initialize_losses(logs, False)
        self.initialize_metrics(logs, False)
        
        gen = self.val_dataloader
        if self.is_main_rank:
            gen = self.progbar(gen, total=len(gen),
                               desc='Validation: ')
        try:
            with torch.no_grad():
                for i, batch in enumerate(gen):
                    self.on_batch_begin(i, logs, False)
                    batch = self.preprocess_batch(batch=batch, training=False)
                    outputs, loss = self.validation_step(batch, logs)
                    
                    self.update_losses(i, loss, batch, logs, False)
                    self.update_metrics(outputs, batch, logs, False)
                    
                    self.on_batch_end(i, logs, False)
                    
                    if self.is_main_rank:
                        desc = 'Validation: '+'; '.join(self.log_description(i, logs, False))
                        gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
    
    def load_history(self):
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        try:
            history = cfg.safe_load(history_file)
            if history is None:
                history = []
                if self.is_main_rank:
                    os.rename(history_file, history_file+'.corrupted')
                    print('Warning: Possibly corrupted history file. Moved to',
                          history_file+'.corrupted')
            return history
        except FileNotFoundError:
            return []
    
    def save_history(self, history):
        os.makedirs(self.config.log_path, exist_ok=True)
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        cfg.safe_dump(history, history_file)
    
    def set_all_lr(self, new_lr):
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def set_lr_from_schedule(self, epoch, logs):
        if self.config.lr_schedule is None: return
        new_lr = self.config.max_lr
        max_epoch = -1
        for e,lr in self.config.lr_schedule:
            if e <= epoch and e > max_epoch:
                new_lr = lr
                max_epoch = e
                
        self.set_all_lr(new_lr)
        logs['lr'] = new_lr
    
    def train_model(self):
        if self.is_main_rank: 
            history = self.load_history()
        starting_epoch = self.state.current_epoch
        
        self.on_train_begin()
        should_stop_training = False
        try:
            for i in range(starting_epoch, self.config.num_epochs):
                timestamps = [datetime.now()]
                self.state.current_epoch = i
                if self.is_main_rank: 
                    print(f'\nEpoch {i+1}/{self.config.num_epochs}:', flush=True)
                logs = dict(epoch = self.state.current_epoch, 
                            global_step = self.state.global_step)
                
                try:
                    self.set_lr_from_schedule(i, logs)
                    self.on_epoch_begin(logs, True)
                    if self.config.training_type == 'normal':
                        self.train_epoch(i, logs)
                    elif self.config.training_type == 'minimal':
                        self.train_epoch(i, logs, minimal=True)
                    else:
                        raise ValueError(f'Unknown training type: {self.config.training_type}')
                    self.on_epoch_end(logs, True)
                except StopTrainingException:
                    should_stop_training = True
                else:
                    if str(logs.get('loss', 0.)) == 'nan':
                        return 'nan'
                    if str(logs.get('loss', 0.)) == 'inf':
                        return 'inf'
                    self.recovery_tries = 0
                timestamps.append(datetime.now())
                
                try:
                    if (self.val_dataloader is not None)\
                            and (not ((i+1) % self.config.validation_frequency)):
                        cond = True if self.config.validation_condition is None \
                                    else eval(self.config.validation_condition, logs.copy())
                        if cond:
                            self.on_epoch_begin(logs, False)
                            if self.config.evaluation_type == 'validation':
                                self.validation_epoch(i, logs)
                            elif self.config.evaluation_type == 'prediction':
                                self.prediction_epoch(i, logs)
                            else:
                                raise ValueError(f'Unknown evaluation type: {self.config.evaluation_type}')
                    self.on_epoch_end(logs, False)
                except StopTrainingException:
                    should_stop_training = True
                timestamps.append(datetime.now())
                
                logs.update(start_time=timestamps[0].strftime('%m/%d-%H:%M:%S'))
                logs.update(train_time=str(timestamps[1]-timestamps[0]))
                logs.update(val_time=str(timestamps[2]-timestamps[1]))
                self.state.current_epoch = i + 1
                if self.is_main_rank:
                    self.save_checkpoint()
                    
                    history.append(logs)
                    self.save_history(history)
                
                if should_stop_training:
                    if self.is_main_rank:
                        print('Stopping training ...')
                    break
        finally:
            self.on_train_end()
    
    def distributed_barrier(self):
        if self.is_distributed:
            dummy = torch.ones((),dtype=torch.int64).cuda()
            torch.distributed.all_reduce(dummy)
    
    # Prediction logic
    def prediction_step(self, batch):
        with self.fwd_pass_context():
            predictions = self.model(batch)
        if isinstance(batch, torch.Tensor):
            return dict(inputs=batch, predictions=predictions)
        elif isinstance(batch, list):
            outputs = batch.copy()
            batch.append(predictions)
            return outputs
        elif isinstance(batch, dict):
            outputs = batch.copy()
            outputs.update(predictions=predictions)
            return outputs
    
    def prediction_loop(self, dataloader,
                        predict_in_train=False):
        if predict_in_train:
            self.model.train()
        else:
            self.model.eval()
        
        outputs = []
        
        if self.is_main_rank:
            gen = self.progbar(dataloader, total=len(dataloader),
                               desc='Prediction: ')
        else:
            gen = dataloader
        try:
            with torch.no_grad():
                for batch in gen:
                    batch = self.preprocess_batch(batch=batch, training=False)
                    outputs.append(self.prediction_step(batch))
        finally:
            if self.is_main_rank: gen.close()
        
        return outputs
    
    def preprocess_predictions(self, outputs):
        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], dict):
            return {k: torch.cat([o[k] for o in outputs], dim=0) 
                    for k in outputs[0].keys()}
        elif isinstance(outputs[0], list):
            return [torch.cat([o[i] for o in outputs], dim=0) 
                    for i in range(len(outputs[0]))]
        else:
            raise ValueError('Unsupported output type')
    
    def postprocess_predictions(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()
        elif isinstance(outputs, dict):
            return {k: v.cpu().numpy() for k, v in outputs.items()}
        elif isinstance(outputs, list):
            return [v.cpu().numpy() for v in outputs]
        else:
            raise ValueError('Unsupported output type')
    
    def distributed_gather_tensor(self, tensors):
        shapes = torch.zeros(self.ddp_world_size+1, dtype=torch.long).cuda()
        shapes[self.ddp_rank+1] = tensors.shape[0]
        torch.distributed.all_reduce(shapes)
        
        offsets = torch.cumsum(shapes, dim=0)
        all_tensors = torch.zeros(offsets[-1], *tensors.shape[1:], dtype=tensors.dtype).cuda()
        all_tensors[offsets[self.ddp_rank]:offsets[self.ddp_rank+1]] = tensors
        
        torch.distributed.all_reduce(all_tensors)
        return all_tensors
    
    def distributed_gather_predictions(self, predictions):
        if self.is_main_rank:
            print('Gathering predictions from all ranks...')
        
        if isinstance(predictions, torch.Tensor):
            all_predictions = self.distributed_gather_tensor(predictions)
        elif isinstance(predictions, list):
            all_predictions = [self.distributed_gather_tensor(pred) for pred in predictions]
        elif isinstance(predictions, dict):
            all_predictions = {key:self.distributed_gather_tensor(pred) 
                               for key, pred in predictions.items()}
        else:
            raise ValueError('Unsupported output type')
        
        if self.is_main_rank:
            print('Done.')
        return all_predictions
    
    def save_predictions(self, dataset_name, predictions):
        os.makedirs(self.config.predictions_path, exist_ok=True)
        predictions_file = os.path.join(self.config.predictions_path, f'{dataset_name}.pt')
        torch.save(predictions, predictions_file)
        print(f'Saved predictions to {predictions_file}')
    
    def evaluate_predictions(self, predictions):
        raise NotImplementedError
    
    def prediction_epoch(self, epoch, logs):
        if self.is_main_rank:
            print(f'Predicting on validation dataset...')
        dataloader = self.val_dataloader
        outputs = self.prediction_loop(dataloader)
        outputs = self.preprocess_predictions(outputs)
    
        if self.is_distributed:
            outputs = self.distributed_gather_predictions(outputs)
        
        predictions = self.postprocess_predictions(outputs)
        if self.is_main_rank:
            self.save_predictions('validation', predictions)
        results = self.evaluate_predictions(predictions)
        for k, v in results.items():
            if isinstance(v, np.ndarray) or isinstance(v, np.number):
                results[k] = v.tolist()
        results = {f'val_{k}': v for k, v in results.items()}
        logs.update(results)
        if self.is_main_rank:
            desc = 'Validation: '+'; '.join(f'{k}: {v:.4f}' for k, v in results.items())
            print(desc, flush=True)
            
    
    # Interface
    def prepare_for_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_checkpoint()
        self.model_summary()
        
    def execute_training(self):
        self.prepare_for_training()
        ending_state = self.train_model()
        self.finalize_training(ending_state)
    
    def finalize_training(self, ending_state):
        if ending_state in ('nan', 'inf'):
            self.recovery_tries += 1
            print(f'{ending_state} loss encountered. '
                  f'Trying to recover for the {self.recovery_tries}\'th time...')
            self.load_checkpoint()
            ending_state = self.train_model()
            self.finalize_training(ending_state)
    
        
