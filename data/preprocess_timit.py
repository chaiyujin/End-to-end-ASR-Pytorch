import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
sys.path.insert(0, '..')


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def read_text(file,target):
    labels = []
    if target == 'phoneme':
        with open(file.replace('.WAV','.PHN'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
    elif target in ['char','subword','word']:
        with open(file.replace('.WAV','.WRD'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
        if target == 'char':
            labels = [c for c in ' '.join(labels)]
    else:
        raise ValueError('Unsupported target: '+target)
    return labels


if __name__ == "__main__":
    from src.preprocess import extract_feature, encode_target
    parser = argparse.ArgumentParser(description='Preprocess program for TIMIT dataset.')
    parser.add_argument('--data_path', type=str, help='Path to raw TIMIT dataset')
    parser.add_argument('--feature_type', default='mfcc', type=str, help='Feature type ( mfcc / fbank )', required=False)
    parser.add_argument('--feature_dim', default=13, type=int, help='Dimension of feature', required=False)
    parser.add_argument('--apply_delta', default=True, type=boolean_string, help='Append Delta', required=False)
    parser.add_argument('--apply_delta_delta', default=True, type=boolean_string, help='Append Delta Delta', required=False)
    parser.add_argument('--apply_cmvn', default=False, type=boolean_string, help='Apply CMVN on feature', required=False)
    parser.add_argument('--output_path', default='.', type=str, help='Path to store output', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--target', default='phoneme', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)
    parser.add_argument('--n_tokens', default=1000, type=int, help='Vocabulary size of target', required=False)
    parser.add_argument('--win_size', default=64, type=int, help='win_size in ms', required=False)
    parser.add_argument('--hop_size', default=8, type=int, help='hop_size in ms', required=False)
    paras = parser.parse_args()

    # check args
    assert not paras.apply_cmvn, "--apply_cmvn should be 'False'"
    assert paras.feature_type in ["mfcc", "fbank"],\
        "unknown feature_type '{}'".format(paras.feature_type)

    # Process training data
    print('')
    print('Preprocessing training data...', end='')
    todo = list(Path(os.path.join(paras.data_path,'train')).rglob("*.[wW][aA][vV]"))
    print(len(todo),'audio files found in training set (should be 4620)')

    print('Extracting acoustic feature...', flush=True)
    tr_x = Parallel(n_jobs=paras.n_jobs)(
        delayed(extract_feature)(
            str(file),
            sr=16000,
            window_size=paras.win_size,
            stride=paras.hop_size,
            feature=paras.feature_type,
            dim=paras.feature_dim,
            cmvn=paras.apply_cmvn,
            delta=paras.apply_delta,
            delta_delta=paras.apply_delta_delta
        )
        for file in tqdm(todo)
    )
    print('Encoding training target...', flush=True)
    tr_y = Parallel(n_jobs=paras.n_jobs)(
        delayed(read_text)(str(file),target=paras.target)
        for file in tqdm(todo)
    )
    tr_y, encode_table = encode_target(tr_y,table=None,mode=paras.target,max_idx=paras.n_tokens)

    # output dir
    dim = paras.feature_dim*(1+paras.apply_delta+paras.apply_delta_delta)
    output_dir = os.path.join(paras.output_path,'_'.join([
        'timit',
        str(paras.feature_type)+str(dim),
        str(paras.target)+str(len(encode_table)),
        "w{}h{}".format(paras.win_size, paras.hop_size)
    ]))

    # save training data
    print('Saving training data to',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir,"train_x.pkl"), "wb") as fp:
        pickle.dump(tr_x, fp)
    del tr_x
    with open(os.path.join(output_dir,"train_y.pkl"), "wb") as fp:
        pickle.dump(tr_y, fp)
    del tr_y
    with open(os.path.join(output_dir,"mapping.pkl"), "wb") as fp:
        pickle.dump(encode_table, fp)

    # Process testing data
    print('Preprocessing testing data...',end='')
    todo = list(Path(os.path.join(paras.data_path,'test')).rglob("*.[wW][aA][vV]"))
    print(len(todo),'audio files found in test set (should be 1680)')

    print('Extracting acoustic feature...',flush=True)
    tt_x = Parallel(n_jobs=paras.n_jobs)(
        delayed(extract_feature)(
            str(file),
            sr=16000,
            window_size=paras.win_size,
            stride=paras.hop_size,
            feature=paras.feature_type,
            dim=paras.feature_dim,
            cmvn=paras.apply_cmvn,
            delta=paras.apply_delta,
            delta_delta=paras.apply_delta_delta
        )
        for file in tqdm(todo)
    )
    print('Encoding testing target...',flush=True)
    tt_y = Parallel(n_jobs=paras.n_jobs)(
        delayed(read_text)(str(file),target=paras.target)
        for file in tqdm(todo)
    )
    tt_y, encode_table = encode_target(tt_y,table=encode_table,mode=paras.target,max_idx=paras.n_tokens)

    print('Saving testing data to',output_dir)
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    with open(os.path.join(output_dir,"test_x.pkl"), "wb") as fp:
        pickle.dump(tt_x, fp)
    del tt_x
    with open(os.path.join(output_dir,"test_y.pkl"), "wb") as fp:
        pickle.dump(tt_y, fp)
    del tt_y
    print('All done, exit.')
