"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
import os
import shutil
import gc
import time
import glob

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config_modified import args
from models.video_net import VideoNet
from data.lrs2_dataset import LRS2Pretrain
from data.utils import collate_fn
from utils.general import num_params, train, evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def curriculum(args):
    PRETRAIN_NUM_WORDS = [1, 2, 3, 5, 7, 9, 11, 13, 17, 21, 29, 37]
    PRETRAIN_CONFIG = {
        1: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        2: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        3: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        5: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        7: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        9: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        11: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        13: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        17: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        21: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        29: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32},
        37: {'PRETRAINED_MODEL_FILE': None, 'BATCH_SIZE': 32}
    }

    # Create parent directory for the checkpoints of this curriculum run
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cp_parent = args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/" + timestr
    os.makedirs(cp_parent, exist_ok=True)

    # Start curriculum learning loop
    for n, num_words in enumerate(PRETRAIN_NUM_WORDS):
        train_over = False
        PRETRAIN_CONFIG[num_words]['CP_DIRECTORY'] = args["CP_DIRECTORY"] + "/" + timestr + "/" + str(num_words)
        PRETRAIN_CONFIG[num_words]['PRETRAIN_NUM_WORDS'] = num_words

        # If we're not at the first step, load pretrained model
        if num_words > 1:
            prev_cp = cp_parent + "/" + str(PRETRAIN_NUM_WORDS[n-1])
            prev_model = glob.glob(prev_cp + "/models/*best*")[0]
            PRETRAIN_CONFIG[num_words]['PRETRAINED_MODEL_FILE'] = prev_model.replace(args['CODE_DIRECTORY'], "")

        while not train_over:
            cfg = args.copy()
            cfg.update(PRETRAIN_CONFIG[num_words])
            try:
                main(args=cfg, data_parallel=(num_words >= 11))
                train_over = True
            except RuntimeError as e:
                print(f"Runtime Error... Trying Again: \n{e}")
                PRETRAIN_CONFIG[num_words]['BATCH_SIZE'] //= 2
                
            torch.cuda.empty_cache()
            gc.collect()


def main(args, data_parallel=False):
    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the pretrain and the preval datasets and the corresponding dataloaders
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    pretrainData = LRS2Pretrain("pretrain", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                videoParams)
    pretrainLoader = DataLoader(pretrainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)
    prevalData = LRS2Pretrain("preval", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                              videoParams)
    prevalLoader = DataLoader(prevalData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)


    #declaring the model, optimizer, scheduler and the loss function
    model = VideoNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                     args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    if data_parallel:
         model = nn.DataParallel(model, device_ids=[0, 1])   #DataParallel
    model.to(device)
    
    # optimizer = optim.SGD(model.parameters(), lr=args["INIT_LR"], momentum=args["MOMENTUM1"]), weight_decay=args["WEIGHT_DECAY"], nesterov=True)    # TODO: Change to SGD with Momentum / Nesterov *LR Multiply by 10
    optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                     patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"],
                                                     threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=False)


    #removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"]):
        # while True:
        #     ch = input("Continue and remove the checkpoints directory - " + args["CP_DIRECTORY"] + " ? y/n: ")
        #     if ch == "y":
        #         break
        #     elif ch == "n":
        #         exit()
        #     else:
        #         print("Invalid input")
        shutil.rmtree(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"])

    os.mkdir(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"])
    os.mkdir(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/models")
    os.mkdir(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/plots")


    #loading the pretrained weights
    if args["PRETRAINED_MODEL_FILE"] is not None:
        print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
        print("\nLoading the pre-trained model .... \n")
        if data_parallel:
            model.module.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device))   # DataParallel
        else:
            model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device))
        model.to(device)
        print("Loading Done.\n")



    trainingLossCurve = list()
    validationLossCurve = list()
    trainingWERCurve = list()
    validationWERCurve = list()


    #printing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" %(numTotalParams))
    print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

    print("Number of Words = %d" %(args["PRETRAIN_NUM_WORDS"]))
    print("\nPretraining the model .... \n")

    trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}
    valParams = {"decodeScheme":"greedy", "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"]}

    for step in range(args["NUM_STEPS"]):

        #train the model for one step
        trainingLoss, trainingCER, trainingWER = train(model, pretrainLoader, optimizer, loss_function, device, trainParams)
        trainingLossCurve.append(trainingLoss)
        trainingWERCurve.append(trainingWER)

        #evaluate the model on validation set
        validationLoss, validationCER, validationWER = evaluate(model, prevalLoader, loss_function, device, valParams)
        validationLossCurve.append(validationLoss)
        validationWERCurve.append(validationWER)

        #printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
              %(step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))

        #make a scheduler step
        scheduler.step(validationWER)

        #saving the model with the lower WER
        if len(validationWERCurve) == 1 or validationWER < min(validationWERCurve[:-1]):
            #remove previous best
            if len(validationWERCurve) > 1:
                os.remove(savePathBest)

            savePathBest = args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/models/pretrain_{:03d}w-step_{:04d}-wer_{:.3f}_best.pt".format(args["PRETRAIN_NUM_WORDS"],
                                                                                                                                            step,
                                                                                                                                            validationWER)
            if data_parallel:
                torch.save(model.module.state_dict(), savePathBest)     # DataParallel
            else:
                torch.save(model.state_dict(), savePathBest)

        #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step%args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]-1)) and (step != 0):

            savePath = args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/models/pretrain_{:03d}w-step_{:04d}-wer_{:.3f}.pt".format(args["PRETRAIN_NUM_WORDS"],
                                                                                                                        step, validationWER)
            if data_parallel:
                torch.save(model.module.state_dict(), savePath)     # DataParallel
            else:
                torch.save(model.state_dict(), savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Loss value")
            plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/plots/pretrain_{:03d}w-step_{:04d}-loss.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()

            plt.figure()
            plt.title("WER Curves")
            plt.xlabel("Step No.")
            plt.ylabel("WER")
            plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + args["CP_DIRECTORY"] + "/plots/pretrain_{:03d}w-step_{:04d}-wer.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()


    print("\nPretraining Done.\n")

    return


if __name__ == "__main__":
    curriculum(args)
