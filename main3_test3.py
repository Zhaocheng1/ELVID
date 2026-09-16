# from SurvivalPredictionModel import SurvivalPredictionModel
from   train_vedio_fusion10_moe_multi_token_expert_09072 import FusionTrainer,FusionTester
from utils.OPTS1 import opts
import fire
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(**kwargs):
    opt = opts().parse()

    SynModel = FusionTrainer(opt=opt)
    SynModel.train()


def test(**kwargs):
    opt = opts().parse()
    SynModel = FusionTester(opt=opt)
    SynModel.test()


if __name__ == '__main__':
    # train()
    test()
    fire.Fire()