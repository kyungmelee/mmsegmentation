# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook

from mmseg.core import DistEvalHook, EvalHook

import pandas as pd 
import matplotlib.pyplot as plt  
import io 

import sys
import os
sys.path.append(os.getcwd()+'\\model-analysis')
from analysis import calculator, calc_metric

@HOOKS.register_module()
class CustomTensorboardLoggerHook(TensorboardLoggerHook):
    """Enhanced Tensorboard logger hook for MMSegmentation.
    """

    def __init__(self,
                 log_dir: str = None,
                 interval=50,
                 ignore_last : bool = True,
                 reset_flag : bool = False,
                 by_epoch : bool = True,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100):
        super().__init__(log_dir, interval, ignore_last, reset_flag, by_epoch)
        
        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = (
            log_checkpoint and log_checkpoint_metadata)
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0)
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None
        self.test_fn = None

    @master_only
    def before_run(self, runner):
        super().before_run(runner)

        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, EvalHook):
                from mmseg.apis import single_gpu_test
                self.eval_hook = hook
                self.test_fn = single_gpu_test
            if isinstance(hook, DistEvalHook):
                from mmseg.apis import multi_gpu_test
                self.eval_hook = hook
                self.test_fn = multi_gpu_test

        # Check conditions to log checkpoint
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log checkpoint in CustomTensorboardLoggerHook, `CheckpointHook` is'
                    'required, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        # Check conditions to log evaluation
        if self.log_evaluation or self.log_checkpoint_metadata:
            if self.eval_hook is None:
                self.log_evaluation = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log evaluation or checkpoint metadata in '
                    'CustomTensorboardLoggerHook, `EvalHook` or `DistEvalHook` in mmseg '
                    'is required, please check whether the validation '
                    'is enabled.')
            else:
                self.eval_interval = self.eval_hook.interval
                self.val_dataset = self.eval_hook.dataloader.dataset
                # Determine the number of samples to be logged.
                if self.num_eval_images > len(self.val_dataset):
                    self.num_eval_images = len(self.val_dataset)
                    runner.logger.warning(
                        f'The num_eval_images ({self.num_eval_images}) is '
                        'greater than the total number of validation samples '
                        f'({len(self.val_dataset)}). The complete validation '
                        'dataset will be logged.')

        # Check conditions to log checkpoint metadata
        if self.log_checkpoint_metadata:
            assert self.ckpt_interval % self.eval_interval == 0, \
                'To log checkpoint metadata in CustomTensorboardLoggerHook, the interval ' \
                f'of checkpoint saving ({self.ckpt_interval}) should be ' \
                'divisible by the interval of evaluation ' \
                f'({self.eval_interval}).'

        # Initialize evaluation table
        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add data to the data table
            self._add_ground_truth(runner)
            # Log ground truth data
            self._log_data_table()

    # for the reason of this double-layered structure, refer to
    # https://github.com/open-mmlab/mmdetection/issues/8145#issuecomment-1345343076
    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super().after_train_iter(runner)
        else:
            super().after_train_iter(runner)
        self._after_train_iter(runner)

    @master_only
    def _after_train_iter(self, runner):
        if self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_iters(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'iter_{runner.iter+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            # Currently the results of eval_hook is not reused by wandb, so
            # wandb will run evaluation again internally. We will consider
            # refactoring this function afterwards
            results = self.test_fn(runner.model, self.eval_hook.dataloader)
            # Initialize evaluation table
            self._init_pred_table()
            # Log predictions
            self._log_predictions(results, runner)
            # Log the table
            self._log_eval_table(runner.iter + 1)

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self): # for ground truth , input 
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'input', 'gt_seg', 'gt_n', 'gt_area', 'gt_edge', 'gt_contrast']
        self.data_table = pd.DataFrame(columns = columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'gt_n', 'pred_seg', 'pred_n', 'pred_idx', 'pred_area', 'result']
        self.eval_table = pd.DataFrame(columns = columns)

    def _add_ground_truth(self, runner):
        # Get image loading pipeline
        from mmseg.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'LoadImageFromFile is required to add images '
                'to W&B Tables.')
            return

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        classes = self.val_dataset.CLASSES
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.img_infos[idx]
            image_name = img_info['filename']
            
            # Get image and convert from BGR to RGB
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=self.val_dataset.img_dir))
            image = mmcv.bgr2rgb(img_meta['img'])

            # Get segmentation mask
            seg_mask = self.val_dataset.get_gt_seg_map_by_idx(idx)
            # Dict of masks to be logged.
            seg_mask_255 = seg_mask * 100 # [0,100,200] scale up 

            if seg_mask.ndim == 2:
                # Log a row to the data table.
                # init info : ['image_name', 'input', 'gt_seg', 'gt_n', 'gt_area', 'gt_edge', 'gt_contrast']
                self.data_table = self.data_table.append({'image_name': image_name, 'input' : image, 'gt_seg' : seg_mask_255, 'gt_n' : -1 , 'gt_area' : -1, 'gt_edge' : -1, 'gt_contrast' : -1}, ignore_index = True)

            else:
                runner.logger.warning(
                    f'The segmentation mask is {seg_mask.ndim}D which '
                    'is not supported by W&B.')
                self.log_evaluation = False
                return

    def _log_predictions(self, results, runner):
        table_idxs = len(self.data_table)
        assert table_idxs == len(self.eval_image_indexs)
        assert len(results) == len(self.val_dataset.img_infos)
        
        from mmseg.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        self.eval_res = []      
        for idx in range(0,len(self.val_dataset.img_infos)):
            img_info = self.val_dataset.img_infos[idx]
            image_name = img_info['filename']
            
            # Get image and convert from BGR to RGB
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=self.val_dataset.img_dir))
            image = mmcv.bgr2rgb(img_meta['img'])

            # Get segmentation mask
            seg_mask = self.val_dataset.get_gt_seg_map_by_idx(idx)
            # Dict of masks to be logged.
            seg_mask_255 = (seg_mask == 2 )* 255 # [0,100,200] scale up 

            pred_mask = (results[idx] ==2 ) * 255

            if seg_mask.ndim == 2:
                # Log a row to the data table.
                result_all_info = calculator(image_name, pred_mask, seg_mask_255)
                # columns = ['image_name', 'gt_n', 'pred_seg', 'pred_n', 'pred_idx', 'pred_area', 'result']
                self.eval_table = self.eval_table.append({'image_name': image_name, 'gt_n' : -1 ,'pred_seg' : pred_mask, 'pred_n' : -1, 'pred_idx' : -1, 'pred_area' : -1, 'result' : 'NONE'}, ignore_index = True)
                self.eval_res = self.eval_res + result_all_info
                
            else:
                runner.logger.warning(
                    'The predictio segmentation mask is '
                    f'{pred_mask.ndim}D which is not supported by W&B.')
                self.log_evaluation = False
                return
        

    def _log_data_table(self):
        #upload sample images 
        for img_idx, data in self.data_table.iterrows():
            tag = f'input image samples num {self.num_eval_images} / {data.image_name} '
            fig = plt.figure(figsize=(2*4,1*4)) # figsize is inches
            plt.rc('font', size=5) # controls default text sizes
            plt.subplot(1, 2, 1, title='input') #, fontsize=10
            plt.grid(False)
            plt.imshow(data.input)
            plt.subplot(1,2, 2, title='gt_scale_up') #, fontsize=10
            plt.imshow(data.gt_seg)
            self.writer.add_figure(tag, fig)
            plt.close(fig)

    def _log_eval_table(self, iter):
        # upload rawdata
        res_df = pd.DataFrame(list(map(lambda p: p.to_dict(), self.eval_res)))
        self.writer.add_text('val/anormal-raw-data',res_df.to_markdown(), iter)
        
        (all_precision, all_recall, precision_array, recall_array, img_num_array, blob_num_array ,TP_array, FP_array, FN_array) = calc_metric(res_df, save=False)

        # with step ->> precision , recall 
        self.writer.add_scalar('val/anormal-precision', all_precision, iter)
        self.writer.add_scalar('val/anormal-recall', all_recall, iter)

        # with step ->> table 
        area_range = [0, 30, 40, 60, 120, 16641]
        result_df = pd.DataFrame(precision_array, index=area_range[0:5])
        self.writer.add_text('val/anormal-precision_group', result_df.to_markdown(), iter)
        result_df = pd.DataFrame(recall_array, index=area_range[0:5])
        self.writer.add_text('val/anormal-recall_group', result_df.to_markdown(), iter)        