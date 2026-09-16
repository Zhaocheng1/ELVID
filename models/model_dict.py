from models.resnet18_based_1 import MultiViewResNetInteraction
from models.resnet18_based_1 import ModifiedResNet18
from models.resnet18_based_avgear import ModifiedResNet18_1
from models.resnet18_based_weiduyasuo import ModifiedResNet18_2
# from models.ablation_spat import MultiViewResNetInteraction_abla_spat
# from models.ablation_tem import MultiViewResNetInteraction_abla_tem
# from models.ablation_backbone import MultiViewResNetInteraction_abla_backbone
def get_model(modelname='resnet18'):
    if modelname == 'resnet18':
        model = ModifiedResNet18()
    elif modelname == '8view':
        model = MultiViewResNetInteraction()
    elif modelname == 'resnet18_avgear':
        model = ModifiedResNet18_1()
    elif modelname == 'resnet18_weiduyasuo':
        model = ModifiedResNet18_2()
    # elif modelname == 'abla_spat':
    #     model = MultiViewResNetInteraction_abla_spat()
    # elif modelname == 'abla_tem':
    #     model = MultiViewResNetInteraction_abla_tem()
    # elif modelname == 'abla_backbone':
    #     model = MultiViewResNetInteraction_abla_backbone()
    else:
        raise RuntimeError("Could not find the model:",modelname)
    return model