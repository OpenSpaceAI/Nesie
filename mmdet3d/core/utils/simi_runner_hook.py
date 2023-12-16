# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, master_only, allreduce_params
from mmcv.fileio import FileClient

@HOOKS.register_module()
class SimiRunnerHook(Hook):
    r"""Exponential Moving Average Hook.
    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.
        .. math::
            Xema\_{t+1} = (1 - \text{momentum}) \times
            Xema\_{t} +  \text{momentum} \times X_t
    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str): The checkpoint path. Defaults to None.
    """

    def __init__(self, 
                resume_from=None,
                load_from=None,
                interval=-1,
                by_epoch=True,
                save_optimizer=True,
                out_dir=None,
                max_keep_ckpts=-1,
                save_last=True,
                sync_buffer=False,
                file_client_args=None,                
                **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args
        self.resume_from = resume_from
        self.load_from = load_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.
        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for hook in model.hooks:
            if hook.__class__.__name__ == 'SimiTeacherHook':
                model.epoch = runner.epoch
                model.lb_list =  runner.lb_list
                model.ulb_list = torch.zeros_like(runner.ulb_list)
                model.lb_map = runner.lb_map
                model.ulb_map = runner.ulb_map
                model.ulb_flag = torch.ones((len(runner.ulb_map)))
                getattr(hook, 'hooks_before_run')(model)
        if self.resume_from  is not None:
            runner.resume(self.resume_from)
        if self.load_from is not None:
            runner.load_checkpoint(self.load_from)
        
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        runner.logger.info((f'ema Checkpoints will be saved to {self.out_dir} by '
                            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink


    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for hook in model.hooks:
            if hook.__class__.__name__ == 'SimiTeacherHook':
                getattr(hook, 'hooks_after_train_iter')(curr_step)
        
        if self.by_epoch:
            return
        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for hook in model.hooks:
            if hook.__class__.__name__ == 'SimiTeacherHook':
                model.epoch = runner.epoch
                getattr(hook, 'hooks_after_train_epoch')(model)
        
        if not self.by_epoch:
            return
        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving ema checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)


    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for hook in model.hooks:
            if hook.__class__.__name__ == 'SimiTeacherHook':
                getattr(hook, 'hooks_before_train_epoch')(model)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner.save_checkpoint(
            self.out_dir, filename_tmpl='epoch_{}_ema.pth', save_optimizer=self.save_optimizer, **self.args)
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}_ema.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}_ema.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt_ema'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}_ema.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}_ema.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break
