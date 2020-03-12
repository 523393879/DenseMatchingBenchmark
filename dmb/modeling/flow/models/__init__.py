from .HMNet import HMNet

_META_ARCHITECTURES = {
    'HMNet': HMNet,
}


def build_flow_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
