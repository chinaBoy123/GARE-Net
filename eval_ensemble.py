import logging
import argparse
from lib import evaluation
from lib.modules import set_seeds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',help='coco or f30k')
    parser.add_argument('--result1', default='/home/ubuntu/Students/zhoutao/code_updated/GARE-Net/runs/runX/checkpoint/coco_gru/results_coco_precomp_i2t.npy')
    parser.add_argument('--result2', default='/home/ubuntu/Students/zhoutao/code_updated/GARE-Net/runs/runX/checkpoint/coco_gru/results_coco_precomp_t2i.npy')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()


    if opt.dataset == 'coco':
        if not opt.evaluate_cxc:
            # Evaluate COCO 5-fold 1K
            # evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=True)
            evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=False)
        else:
            # Evaluate COCO-trained models on CxC
            evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
    elif opt.dataset == 'f30k':
        # Evaluate Flickr30K
        evaluation.eval_ensemble(results_paths=[opt.result1,opt.result2], fold5=False)


if __name__ == '__main__':
    main()
