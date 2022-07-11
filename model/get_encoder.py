import sys
sys.path.append('..')
import torch
from model.encoder import swin_transformer,simplenet,resnet,mobilenetv3,xception,efficientnet


ssl_weight_path = {
    'resnet18':'/staff/shijun/torch_projects/Med_Seg/mae/convert_ckpt/CNN/mae_res18_p8/v2.0-HaN-roi-75/convert_epoch=0399-loss=0.282656.pth.tar',
    'resnet50':'/staff/shijun/torch_projects/Med_Seg/mae/convert_ckpt/CNN/mae_res50_p8/v5.0-HaN-roi-75/convert_epoch=0399-loss=0.266160.pth.tar',
}

def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](classification=False,pretrained=False or weights=='imagenet',**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('simplenet'):
        backbone = simplenet.__dict__[arch](**kwargs)
    elif arch.startswith('mobilenetv3'):
        backbone = mobilenetv3.__dict__[arch](**kwargs)
    elif arch.startswith('xception'):
        backbone = xception.__dict__[arch](**kwargs)
    elif arch.startswith('timm_efficientnet'):
        backbone = efficientnet.__dict__[arch](pretrained=False or weights=='imagenet',**kwargs)
    
    else:
        raise Exception('Architecture undefined!')

    if weights == 'ssl' and isinstance(ssl_weight_path[arch], str):
        print('Loading weights for backbone')
        msg = backbone.load_state_dict(
            torch.load(ssl_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        # if arch.startswith('resnet'):
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(">>>> loaded pre-trained model '{}' ".format(ssl_weight_path[arch]))
        print('>>>> missing keys:',msg[0])
    
    return backbone



if __name__ == '__main__':

    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # net = build_encoder('swin_transformer',n_channels=1)
    # net = build_encoder('resnet18',n_channels=1)
    # net = build_encoder('simplenet',n_channels=1)
    net = build_encoder('mobilenetv3_large_075',n_channels=1)
    # net = build_encoder('xception',n_channels=1)
    # net = build_encoder('timm_efficientnet_b5',n_channels=1, weights='imagenet')
    
    net = net.cuda()
    net.train()
    input = torch.randn((1,1,512,512)).cuda()
    output = net(input)
    for item in output:
        print(item.size())

    import sys
    sys.path.append('..')
    from utils import count_params_and_macs
    count_params_and_macs(net.cuda(),(1,1,512,512))