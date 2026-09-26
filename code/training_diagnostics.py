import torch


def add_structure_loss(prediction, reconstruction, metrics, model_config):
    if 'structure_loss' not in prediction:
        return reconstruction, metrics
    if model_config.get('anchor_fusion') == 'geometry_transformer':
        anatomy = model_config['multiview_transformer']['anatomy']
    else:
        anatomy = model_config['anatomy']
    weight = anatomy.get('loss_weight', .01)
    auxiliary = prediction['structure_loss']
    total = reconstruction + weight * auxiliary
    metrics = dict(metrics, L_anatomy=auxiliary.detach(), anatomy_weight=weight,
                   L_anatomy_weighted=(weight * auxiliary).detach(), loss=total.detach())
    return total, metrics


def transformer_gradient_norms(model):
    groups = {'cross_attention': [], 'spatial_attention': [], 'point_fusion': [], 'structure_head': []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith('point_view_fusion.'):
            groups['point_fusion'].append(parameter.grad)
        elif '.spatial_attention.' in name:
            groups['spatial_attention'].append(parameter.grad)
        elif '.cross_attention.' in name:
            groups['cross_attention'].append(parameter.grad)
        elif '.structure_head.' in name:
            groups['structure_head'].append(parameter.grad)
    return {f'gradient/{name}': float(torch.stack([gradient.detach().float().square().sum() for gradient in gradients]).sum().sqrt())
            for name, gradients in groups.items() if gradients}
