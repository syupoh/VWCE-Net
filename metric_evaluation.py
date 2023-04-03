from numpy import FPE_DIVIDEBYZERO
from tqdm import tqdm
import argparse
import pdb
import json
import os
import numpy as np

from syslogger.syslogger import Logger
import inspect
import sys
from datetime import datetime

import shutil
        

def main_worker(jsonname, number):
    predjson_set = json.load(open(jsonname))
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    acc_true = 0
    if number == 0:
        # total_label = [0, 0, 0, 0, 0]

        # for predjson in predjson_set:
        #     total_label[predjson['label']] += 1
        

        print('{0}'.format(os.path.dirname(jsonname)))
        # print('total_data\t{0}'.format(
        #     len(predjson_set)
        # ), end='\n')
        # print('each_label', end='\t')
        # for each_label in total_label:
        #     print('{0}'.format(each_label), end='\t')
        # print()

        print('\tsens\tspec\taccu\tf1\tcls', end='\t')
        print('TP\tTN\tFN\tFP\tAll')

    for predjson in predjson_set:
        pred_argmax = np.array(predjson['pred']).argmax()
        if  pred_argmax == predjson['label']:
            acc_true += 1 

        pred_argmax = 1 if pred_argmax > 0 else 0
        label = 1 if predjson['label'] > 0 else 0

        if label == 0:
            if label == pred_argmax:
                TN += 1
            else:
                FP += 1
            
        else:
            if label == pred_argmax:
                TP += 1
            else:
                FN += 1

    clsacc = acc_true / len(predjson_set) * 100
    if not TP + FP == 0:
        precision = (TP/(TP + FP)) * 100
    else :
        precision = 0

    if not TP + FN == 0:
        recall = (TP/(TP + FN)) * 100
    else :
        recall = 0

    if not precision + recall == 0:
        f1score = 2 * (precision * recall)/(precision + recall)
    else :
        f1score = 0
    
    if not (TP + TN + FP + FN) == 0:
        accuracy = (TP + TN)/(TP + TN + FP + FN) * 100
    else:
        accuracy = 0
        
    if not (TN + FP) == 0:
        specificity = TN / (TN + FP) * 100
    else:
        specificity = 0


    print(os.path.basename(jsonname), end='')
    print('\t{recall:.02f}\t{specificity:.02f}\t{accuracy:.02f}\t{f1score:.02f}\t{cls:.02f}'.format(
        recall=recall, specificity=specificity, accuracy=accuracy, f1score=f1score, cls=clsacc
    ), end='\t')
    print('{TP}\t{TN}\t{FN}\t{FP}\t{all}'.format(
        TP=TP, TN=TN, FN=FN, FP=FP, all=TP+TN+FN+FP
    ))

  
def get_args():
    parser = argparse.ArgumentParser(description="metric evaluation mmaction")
    
    parser.add_argument(
        '--evalpath', type=str,
        default=''
    )
    parser.add_argument(
        '--dataname', type=str,
        # default='220204_normal_timesformer',
        default='220209_test_2',
    )
    parser.add_argument(
        '--prefix', type=str,
        # default='220204_normal_timesformer',
        default='220204_normal_timesformer',
    )
    parser.add_argument(
        '--root', type=str,
        default='./work_dirs/_220204'
    )
    
    # parser.add_argument(
    #     '--imagesave-thr', type=float,
    #     default=0)
    # parser.add_argument(
    #     '--dataname', type=str,
    #     default='')
    # parser.add_argument(
    #     '--setname', type=str,
    #     default='')

    # parser.add_argument('--vascular-bleeding', action='store_true')
    # parser.add_argument('--all', action='store_true')
    # parser.add_argument('--polyp', action='store_true')
    # parser.add_argument(
    #     '--seqweighted', type=int,
    #     default=0)
    # parser.add_argument(
    #     '--window', type=int,
    #     default=3)
    # parser.add_argument('--imagesave', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    now = datetime.now()
    curtime = now.isoformat()
    curtime = curtime.replace('-', '')
    curtime = curtime.replace(':', '')[2:13]

    # args.prefix = '220204_normal_timesformer'
    # args.root = './work_dirs
    # root = args.root
    # args.dataname  
    if args.evalpath == '':
        root = args.root
        prefix = args.prefix
        dataname = '' if args.dataname == '' else '_' + args.dataname
        outname = f'./metric_log/{prefix}{dataname}_{curtime}.txt'
        sys.stdout = Logger(outname)
        print(outname)

        for foldername in sorted(os.listdir(root)):
            if foldername.startswith(prefix):
                evalpath = os.path.join(root, foldername)

                number = 0 
                for filename in sorted(os.listdir(evalpath)):
                       
                    if args.dataname == '':
                        target_path = os.path.join(evalpath, filename)
                    elif filename.startswith('epoch') and foldername.startswith(args.dataname):
                        target_path = os.path.join(evalpath, filename)
                    elif args.dataname in filename:
                        target_path = os.path.join(evalpath, filename)
                    else:
                        target_path = ''

                    if not filename.endswith('.log.json') and filename.endswith('_pred.json') and target_path != '':
                        main_worker(target_path, number)
                        number += 1

                if number > 0:
                    print()
    else:
        sys.stdout = Logger('{0}/all_evaluation_{1}.txt'.format(
            args.evalpath, curtime)
        )
        number = 0 
        for filename in sorted(os.listdir(args.evalpath)):
            if not filename.endswith('.log.json') and filename.endswith('.json'):
                main_worker(os.path.join(args.evalpath, filename), number)
                number += 1


if __name__ == '__main__':
    main()