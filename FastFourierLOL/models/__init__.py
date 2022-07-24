import logging
logger = logging.getLogger('base')


def create_model(opt):
    # image restoration
    model = opt['model']
    if model == 'sr':
        from .SIEN_model import SIEN_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

