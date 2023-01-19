# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        
        t_idx = 0
        if runner.logger.level >= 10 : #DEBUG
            result_list = []
            if hasattr(self.dataloader.dataset, 'img_infos'):
                for data_info in self.dataloader.dataset.img_infos:
                    gt_image = data_info['ann']['seg_map'] #file_name
                    result_dict = dict(input_name = data_info['filename'], ann_name = gt_image, key_score = key_score)
                    # gt_label2 = gt_label[0] # base 0
                    if key_score < 0.9 :
                        folder_name = '/iter_' + str(runner.iter) + '_false' 
                        self._save_debug_data(data_info, runner.iter, key_score, folder_name, self.dataloader.dataset.img_dir, self.dataloader.dataset.ann_dir)
                    else :
                        folder_name = '/iter_' + str(runner.iter) + '_true'
                        self._save_debug_data(data_info, runner.iter, key_score, folder_name, self.dataloader.dataset.img_dir, self.dataloader.dataset.ann_dir)
                    result_list.append(result_dict)
                    t_idx = t_idx+1

            #save file 
            import csv 
            with open(self.out_dir + '/iter_' + str(runner.iter) + '_pred.csv','a',newline='') as f :
                    writer = csv.writer(f)
                    writer.writerow(result_list[0].keys())
                    for item in result_list:
                        writer.writerow(item.values())

        if self.save_best:
            self._save_ckpt(runner, key_score)

    def _save_debug_data(self, data_info:list, iter :int, results_pred , folder_name: str , input_folder : str, ann_folder : str ) :
        import os , shutil , cv2
        #input path = data_info.img_prefix + '\' + data_info.img_info.filename
        #dst path = log folder + iter index + label 
        input_filename = input_folder + '/' + data_info['filename']
        ann_filename = ann_folder + '/' + data_info['ann']['seg_map']
        savefolder =  self.out_dir + folder_name 
        fileName = os.path.splitext(data_info['filename'])
        savefilepath = savefolder + '/' + fileName[0] + '_score_'+ str(round(results_pred,3)) + '_input' + fileName[1]
        updatedfolder = os.path.dirname(savefilepath)
        if(os.path.isdir(updatedfolder) == False):
            os.makedirs(updatedfolder) #mkdir
        shutil.copyfile(input_filename,savefilepath)
        
        savefilepath = savefolder + '/' + fileName[0] + '_score_'+ str(round(results_pred,3)) + '_groundtruth' + fileName[1]
        ann_image = cv2.imread(ann_filename) 
        ann_image = ann_image * 100 
        cv2.imwrite(savefilepath, ann_image)
        # shutil.copyfile(ann_filename,savefilepath)

class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
