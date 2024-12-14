import sys
sys.path.append("src/function_help")
from import_library_and_function import *
from metric import *
from SNMF_with_LR_loss import *

# main fuction
def main():
    parser = argparse.ArgumentParser(description='Generate an image matrix.')
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the feature file')
    parser.add_argument('--meta_path', type=str, required=True, help='Path to the meta file')
    parser.add_argument('--nmf_init_mode', type=str, required=True, help='How to initialize SNMF')
    parser.add_argument('--rank', type=int, required=True, help='Number of dims of transformed data')
    parser.add_argument('--iter', type=int, required=True, help='Number of iteractions')
    parser.add_argument('--tolerance', type=float, required=True, help='Tolerance in early stopping')
    parser.add_argument('--patience', type=int, required=True, help='Patience in early stopping')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha in ADADELTA')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon in ADADELTA')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the SNMF output directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    feature_path = args.feature_path
    meta_path = args.meta_path 
    nmf_init_mode = args.nmf_init_mode
    rank = args.rank
    iter = args.iter 
    tolerance = args.tolerance
    patience = args.patience
    alpha = args.alpha
    epsilon = args.epsilon
    input_path = args.input_path
    output_path = args.output_path


    # main ###################################################################################################
    epsStab = 2*2.220446049250313e-16

    os.system('mkdir -p {}'.format(output_path))

    # read meta
    meta = pd.read_csv(meta_path)
    meta_train = meta[meta['Set'] == 'train']
    meta['Set'] = meta['SampleID']

    # read feature
    feature = pd.read_csv(feature_path)
    feature = pd.merge(meta[['SampleID']], feature)

    # set list
    set_data_list = meta['Set'].unique().tolist()

    # prepare to transform
    dict_meta = {}
    dict_X = {}

    for set_data in set_data_list:
        
        meta_set_data = meta[meta['Set'] == set_data].reset_index(drop = True)

        X_set_data = pd.merge(meta_set_data[['SampleID']], feature).drop('SampleID', axis = 1)

        print(set_data, X_set_data.shape)
        
        dict_meta[set_data] = meta_set_data.copy()
        dict_X[set_data] = X_set_data.copy()

    # get SNMF info
    with open('{}/summary.pkl'.format(input_path), 'rb') as file:
        # load the data to the file
        summary = pickle.load(file)
        
        
    # SNMF transforming
    tolerance_sample = tolerance / meta_train.shape[0]

    # get X ~ H and weight of LR model
    X = summary['X'].copy()
    weight = summary['weight'].copy()

    # log
    sample_list = []
    loss_list_sample = []
    loss_list1_sample = []
    loss_list2_sample = []

    # loop samples
    for set_data in set_data_list:
        
        # feature
        Y = dict_X[set_data].copy().values

        # transform
        K, loss_list, loss_list1, loss_list2, best_i = SNMF_transform_sample(Y, X, iter, tolerance_sample, patience, epsStab, alpha, epsilon)

        # save log
        sample_list.append(set_data)
        loss_list_sample.append(loss_list[best_i])
        loss_list1_sample.append(loss_list1[best_i])
        loss_list2_sample.append(loss_list2[best_i])
        
        # save transformed data
        dict_X[set_data] = pd.DataFrame(K).copy()
        
    # save transformed data as csv
    df  = pd.DataFrame()

    for set_data in set_data_list:
        
        sample_df = dict_meta[set_data][['SampleID']]
        sample_df.reset_index(drop = True, inplace = True)
        dict_X[set_data].reset_index(drop = True, inplace = True)
        
        data = pd.concat([sample_df, dict_X[set_data]], axis = 1)

        df = pd.concat([df, data], axis = 0)

    df.to_csv('{}/feature.csv'.format(output_path), index = None)

    summary['sample_list_final'] = sample_list
    summary['loss_list_sample_final'] = loss_list_sample

    # open a file in binary write mode
    with open('{}/summary_final.pkl'.format(output_path), 'wb') as file:
        # save the data to the file
        pickle.dump(summary, file)

if __name__ == '__main__':
    main()