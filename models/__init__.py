from .resnet import resnet18

def generate_net(args):
    if not args.model_type in globals().keys():
        raise NotImplementedError("there has no %s" % (args.model_type))

    return globals()[args.model_type](args)

__all__ = ['generate_net']
