import os
import argparse
import logging
from plistlib import InvalidFileException
from lib import evaluation
from lib.modules import set_seeds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco_precomp',
                        help='coco or f30k')
    parser.add_argument('--model_path', default='/home/ubuntu/Students/zhoutao/code_updated/GARE-Net/runs/runX/checkpoint/coco_bert/model_best_t2i.pth')
    parser.add_argument('--data_path', default='/home/ubuntu/Students/zhoutao/data/')
    parser.add_argument('--save_results', action='store_false') # store_true
    parser.add_argument('--evaluate_cxc', action='store_true')
    parser.add_argument('--seed', default=2022, type=int, help='random seed')
    opt = parser.parse_args()

    set_seeds(opt.seed)

    if not os.path.exists(opt.model_path):
        logger.info("Model path '%s' does not exist"%(opt.model_path))
        raise InvalidFileException
    if opt.save_results:  # Save the final results for computing ensemble results
        save_path = os.path.join(os.path.dirname(opt.model_path), 'results_{}.npy'.format(opt.dataset))
    else:
        save_path = None

    if opt.dataset == 'coco_precomp':
        if not opt.evaluate_cxc:
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=True)
            # Evaluate COCO 5K
            # evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
            # evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            # evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=True)
        # else:
            # Evaluate COCO-trained models on CxC
            # evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=True, cxc=True)
    elif opt.dataset == 'f30k_precomp':
        # Evaluate Flickr30K
        evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()
