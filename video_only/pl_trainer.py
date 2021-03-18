import time
import os
import glob
import gc

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
from torch.utils.data import DataLoader

from config_modified import args
from models.video_net import VideoNet
from data.lrs2_dataset import LRS2Pretrain, LRS2Main
from data.utils import collate_fn
from utils.metrics import compute_cer, compute_wer
from utils.decoders import ctc_greedy_decode, ctc_search_decode


class VideoNetDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        # TODO: Change cfg to regular argument names
        super().__init__()
        self.data_cfg = data_cfg
        self.videoParams = {"videoFPS": self.data_cfg["VIDEO_FPS"]}
        self.gpuAvailable = torch.cuda.is_available()
        self.data_cls = LRS2Pretrain if self.data_cfg["PRETRAIN"] else LRS2Main

        if self.data_cfg["PRETRAIN"]:
            self.trainData = LRS2Pretrain("pretrain",
                                          self.data_cfg["DATA_DIRECTORY"],
                                          self.data_cfg["PRETRAIN_NUM_WORDS"],
                                          self.data_cfg["CHAR_TO_INDEX"],
                                          self.data_cfg["STEP_SIZE"],
                                          self.videoParams)
            self.valData = LRS2Pretrain("preval",
                                        self.data_cfg["DATA_DIRECTORY"],
                                        self.data_cfg["PRETRAIN_NUM_WORDS"],
                                        self.data_cfg["CHAR_TO_INDEX"],
                                        self.data_cfg["STEP_SIZE"],
                                        self.videoParams)
        else:
            self.trainData = LRS2Main("train",
                                      self.data_cfg["DATA_DIRECTORY"],
                                      self.data_cfg["MAIN_REQ_INPUT_LENGTH"],
                                      self.data_cfg["CHAR_TO_INDEX"],
                                      self.data_cfg["STEP_SIZE"],
                                      self.videoParams)
            self.valData = LRS2Main("val",
                                    self.data_cfg["DATA_DIRECTORY"],
                                    self.data_cfg["MAIN_REQ_INPUT_LENGTH"],
                                    self.data_cfg["CHAR_TO_INDEX"],
                                    self.data_cfg["STEP_SIZE"],
                                    self.videoParams)

    def train_dataloader(self) -> DataLoader:
        kwargs = {"num_workers": self.data_cfg["NUM_WORKERS"], "pin_memory": True} if self.gpuAvailable else {}
        trainLoader = DataLoader(self.trainData,
                                 batch_size=self.data_cfg["BATCH_SIZE"],
                                 collate_fn=collate_fn,
                                 shuffle=True,
                                 **kwargs)
        return trainLoader

    def val_dataloader(self) -> DataLoader:
        kwargs = {"num_workers": self.data_cfg["NUM_WORKERS"], "pin_memory": True} if self.gpuAvailable else {}
        valLoader = DataLoader(self.valData,
                               batch_size=self.data_cfg["BATCH_SIZE"],
                               collate_fn=collate_fn,
                               shuffle=True,
                               **kwargs)
        return valLoader


class VideoNetPL(pl.LightningModule):
    def __init__(self, net_class, net_cfg, train_cfg):
        super().__init__()
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=False)

        self.model = net_class(**net_cfg)

    def forward(self, inputBatch):
        outputBatch = self.model(inputBatch)
        return outputBatch

    def training_step(self, batch, batch_idx):
        trainParams = {"spaceIx": args["CHAR_TO_INDEX"][" "],
                       "eosIx": args["CHAR_TO_INDEX"]["<EOS>"]}

        inputBatch, targetBatch, inputLenBatch, targetLenBatch = batch
        inputBatch, targetBatch = inputBatch.float(), targetBatch.int()
        inputLenBatch, targetLenBatch = inputLenBatch.int(), targetLenBatch.int()

        outputBatch = self.model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = self.loss_fn(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

        trainingLoss = loss
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(),
                                                                inputLenBatch,
                                                                trainParams["eosIx"])
        trainingCER = compute_cer(predictionBatch,
                                  targetBatch,
                                  predictionLenBatch,
                                  targetLenBatch)
        trainingWER = compute_wer(predictionBatch,
                                  targetBatch,
                                  predictionLenBatch,
                                  targetLenBatch,
                                  trainParams["spaceIx"])

        self.log('train_loss', trainingLoss, prog_bar=True)
        self.log('train_wer', trainingWER, prog_bar=True)
        self.log('train_cer', trainingCER, prog_bar=True)
        return trainingLoss
    
    def validation_step(self, batch, batch_idx):
        evalParams = {"decodeScheme": "greedy",
                      "spaceIx": args["CHAR_TO_INDEX"][" "],
                      "eosIx": args["CHAR_TO_INDEX"]["<EOS>"]}

        inputBatch, targetBatch, inputLenBatch, targetLenBatch = batch
        inputBatch, targetBatch = inputBatch.float(), targetBatch.int()
        inputLenBatch, targetLenBatch = inputLenBatch.int(), targetLenBatch.int()

        outputBatch = self.model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = self.loss_fn(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

        evalLoss = loss
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch,
                                                                    inputLenBatch,
                                                                    evalParams["eosIx"])
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch,
                                                                    inputLenBatch,
                                                                    evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"],
                                                                    evalParams["eosIx"],
                                                                    evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = compute_cer(predictionBatch,
                              targetBatch,
                              predictionLenBatch,
                              targetLenBatch)
        evalWER = compute_wer(predictionBatch,
                              targetBatch,
                              predictionLenBatch,
                              targetLenBatch,
                              evalParams["spaceIx"])

        self.log('val_loss', evalLoss, prog_bar=True)
        self.log('val_wer', evalWER, prog_bar=True)
        self.log('val_cer', evalCER, prog_bar=True)
        return evalLoss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.train_cfg["INIT_LR"],
                               betas=(self.train_cfg["MOMENTUM1"], self.train_cfg["MOMENTUM2"]))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode="min",
                                                         factor=self.train_cfg["LR_SCHEDULER_FACTOR"],
                                                         patience=self.train_cfg["LR_SCHEDULER_WAIT"],
                                                         threshold=self.train_cfg["LR_SCHEDULER_THRESH"],
                                                         threshold_mode="abs",
                                                         min_lr=self.train_cfg["FINAL_LR"],
                                                         verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_wer"
        }


def train_step(args, timestr='', best_ckpt=None):
    data_cfg = {
        "VIDEO_FPS": args["VIDEO_FPS"],
        "DATA_DIRECTORY": args["DATA_DIRECTORY"],
        "PRETRAIN_NUM_WORDS": args["PRETRAIN_NUM_WORDS"],
        "CHAR_TO_INDEX": args["CHAR_TO_INDEX"],
        "STEP_SIZE": args["STEP_SIZE"],
        "NUM_WORKERS": args["NUM_WORKERS"],
        "BATCH_SIZE": args["BATCH_SIZE"],
        "PRETRAIN": args["PRETRAIN"]
    }
    train_cfg = {
        "INIT_LR": args["INIT_LR"],
        "MOMENTUM1": args["MOMENTUM1"],
        "MOMENTUM2": args["MOMENTUM2"],
        "LR_SCHEDULER_FACTOR": args["LR_SCHEDULER_FACTOR"],
        "LR_SCHEDULER_WAIT": args["LR_SCHEDULER_WAIT"],
        "LR_SCHEDULER_THRESH": args["LR_SCHEDULER_THRESH"],
        "FINAL_LR": args["FINAL_LR"],
    }
    net_cfg = {
        "dModel": args["TX_NUM_FEATURES"],
        "nHeads": args["TX_ATTENTION_HEADS"],
        "numLayers": args["TX_NUM_LAYERS"],
        "peMaxLen": args["PE_MAX_LENGTH"],
        "fcHiddenSize": args["TX_FEEDFORWARD_DIM"],
        "dropout": args["TX_DROPOUT"],
        "numClasses": args["NUM_CLASSES"]
    }

    logger = pl_loggers.NeptuneLogger(
        project_name='benso/deep-avsr',
        experiment_name=f'video_only_curriculum',
        params=args,
        tags={'start_date': timestr}
    )

    model_checkpoint = pl_callbacks.ModelCheckpoint(
        filename=args["NUM_WORDS"] + '/{epoch:02d}-{val_wer:.2f}',
        save_weights_only=True,
        save_top_k=3,
        monitor='val_wer',
        period=1
    )

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=model_checkpoint,
        gpus=2,
        auto_select_gpus=False,
        max_epochs=args["NUM_STEPS"],
        accelerator=args["ACCELERATOR"],
        resume_from_checkpoint=best_ckpt
    )

    data = VideoNetDataModule(data_cfg=data_cfg)
    network = VideoNetPL(net_class=VideoNet, net_cfg=net_cfg, train_cfg=train_cfg)
    trainer.fit(model=network, datamodule=data)

    return model_checkpoint.best_model_path


def curriculum(args):
    PRETRAIN_NUM_WORDS = [1, 2, 3, 5, 7, 9, 11, 13, 17, 21, 29, 37, 0]
    PRETRAIN_CONFIG = {
        1: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 1, 'BATCH_SIZE': 32,},
        2: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 2, 'BATCH_SIZE': 32},
        3: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 3, 'BATCH_SIZE': 32},
        5: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 5, 'BATCH_SIZE': 32},
        7: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 7, 'BATCH_SIZE': 32},
        9: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 9, 'BATCH_SIZE': 32},
        11: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 11, 'BATCH_SIZE': 32},
        13: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 13, 'BATCH_SIZE': 32},
        17: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 17, 'BATCH_SIZE': 32},
        21: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 21, 'BATCH_SIZE': 32},
        29: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 29, 'BATCH_SIZE': 32},
        37: {'PRETRAIN': True, 'PRETRAIN_NUM_WORDS': 37, 'BATCH_SIZE': 32},
        0: {'PRETRAIN': False, 'PRETRAIN_NUM_WORDS': 0, 'BATCH_SIZE': 32},
    }

    # Create parent directory for the checkpoints of this curriculum run
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Start curriculum learning loop
    best_ckpt = None
    for n, num_words in enumerate(PRETRAIN_NUM_WORDS):
        train_over = False

        while not train_over:
            cfg = args.copy()
            cfg.update(PRETRAIN_CONFIG[num_words])
            try:
                best_ckpt = train_step(args=cfg, timestr=timestr, best_ckpt=best_ckpt)
                train_over = True
            except RuntimeError as e:
                print(f"Runtime Error... Trying Again: \n{e}")
                PRETRAIN_CONFIG[num_words]['BATCH_SIZE'] //= 2

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    curriculum(args)