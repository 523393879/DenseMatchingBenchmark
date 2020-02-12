from .deep_prunner import DeepPrunerSampler

SAMPLER = {
    "DEEPPRUNER": DeepPrunerSampler,
}


def build_disp_sampler(cfg):
    sampler_type = cfg.model.disp_sampler.type
    assert sampler_type in SAMPLER, "disp_sampler type not found, expected: {}," \
                                    "but got {}".format(SAMPLER.keys(), sampler_type)

    default_args = cfg.model.disp_sampler
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    sampler = SAMPLER[sampler_type](**default_args)

    return sampler
