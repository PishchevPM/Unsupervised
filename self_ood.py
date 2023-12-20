from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from self_ood.data import CIFAR4vs6
from self_ood.models import SelfOOD


data_root = Path('./')
exp_root = Path('./experiments/self_ood_temp/')

batch_size = 512
lr = 1e-2 * batch_size / 256  # change lr proportionally to batch size
num_epochs = 200

datamodule = CIFAR4vs6(
    data_dir=data_root / 'cifar10',
    batch_size=batch_size,
    num_workers=8
)

model = SelfOOD(
    encoder_architecture='resnet18_32x32',
    lr=lr
)


callbacks = [
    LearningRateMonitor()
]

logger = TensorBoardLogger(
    save_dir=exp_root,
    name=f'pretrain/cifar10/vicreg/'
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    accelerator='gpu',
    devices=1,
    max_epochs=num_epochs,
    gradient_clip_val=1.0,
    log_every_n_steps=10
)

trainer.fit(
    model=model,
    datamodule=datamodule
)
