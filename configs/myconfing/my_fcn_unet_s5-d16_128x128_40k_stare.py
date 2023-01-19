# python tools/train.py configs/myconfing/my_fcn_unet_s5-d16_128x128_40k_stare.py --work-dir=./work_dirs/segtestfirst
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', './my_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))

evaluation = dict(metric='mDice')
runner = dict(type='IterBasedRunner', max_iters=800000)
# runner = dict(type='EpochBasedRunner', max_epochs=100) 

log_level = 'DEBUG'