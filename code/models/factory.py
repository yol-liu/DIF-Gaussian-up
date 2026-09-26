"""Keep deterministic model construction independent of probability imports."""

from models.model import DIF_Gaussian


def build_model(cfg):
    if 'probabilistic' in cfg:
        from probabilistic_config import resolve_probability, configure_normalization
        probability = resolve_probability(cfg['probabilistic'])
        if probability['mode'] != 'deterministic':
            from models.dense_probability import DenseProbabilisticDIFGaussian
            return DenseProbabilisticDIFGaussian(cfg)
        model = DIF_Gaussian(cfg)
        configure_normalization(model, probability['normalization'])
        return model
    if cfg.get('latent', {}).get('use_latent', False):
        from models.probabilistic_model import ProbabilisticDIFGaussian
        return ProbabilisticDIFGaussian(cfg)
    return DIF_Gaussian(cfg)
