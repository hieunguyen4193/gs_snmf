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
    parser.add_argument('--meta_path_val', type=str, required=True, help='Path to the validation meta file')
    parser.add_argument('--nmf_init_mode', type=str, required=True, help='How to initialize SNMF')
    parser.add_argument('--rank', type=int, required=True, help='Number of dims of transformed data')
    parser.add_argument('--iter', type=int, required=True, help='Number of iteractions')
    parser.add_argument('--tolerance', type=float, required=True, help='Tolerance in early stopping')
    parser.add_argument('--patience', type=int, required=True, help='Patience in early stopping')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha in ADADELTA')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon in ADADELTA')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    feature_path = args.feature_path
    meta_path = args.meta_path 
    meta_path_val = args.meta_path_val
    nmf_init_mode = args.nmf_init_mode
    rank = args.rank
    iter = args.iter 
    tolerance = args.tolerance
    patience = args.patience
    alpha = args.alpha
    epsilon = args.epsilon
    output_path = args.output_path


    # main ###################################################################################################
    epsStab = 2*2.220446049250313e-16

    os.system('mkdir -p {}'.format(output_path))

    if os.path.exists(f'{output_path}/summary.pkl') == True:
        print('Trained => SKIP')
        sys.exit()

    # read meta
    meta = pd.read_csv(meta_path)
    meta_val = pd.read_csv(meta_path_val)

    # how to encode labels
    dict_encode = {}
    c = 0
    for label in meta['Label'].unique():
        dict_encode[label] = c
        c += 1

    # read feature
    feature = pd.read_csv(feature_path)
    feature = pd.merge(meta[['SampleID']], feature)
    feature_val = pd.read_csv(feature_path)
    feature_val = pd.merge(meta_val[['SampleID']], feature)

    # prepare X ~features, y ~ labels, sample_list
    X_train = pd.merge(meta[['SampleID']], feature)
    y_train = meta['Label'].apply(lambda x: dict_encode[x])
    X_val = pd.merge(meta_val[['SampleID']], feature_val)
    y_val = meta_val['Label'].apply(lambda x: dict_encode[x])


    sample_train = X_train[['SampleID']]
    sample_val = X_val[['SampleID']]
    X_train.drop('SampleID', axis = 1, inplace = True)
    X_val.drop('SampleID', axis = 1, inplace = True)

    sample_train.reset_index(drop = True, inplace = True)
    sample_val.reset_index(drop = True, inplace = True)
    X_train.reset_index(drop = True, inplace = True)
    X_val.reset_index(drop = True, inplace = True)
    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)

    # SNMF training
    summary = {}

    max_ylim = 0
        
    # feature
    Y = X_train.copy().values

    # label
    u = y_train.copy().values

    # train
    X, loss_list, loss_list1, loss_list2, weight, best_i = SNMF(Y, u, rank, iter, tolerance, patience, epsStab, alpha, epsilon, nmf_init_mode)

    # save result
    summary['X'] = X
    summary['weight'] = weight
    summary['best_i'] = best_i
    summary['loss'] = loss_list
    summary['loss1'] = loss_list1
    summary['loss2'] = loss_list2

    # # history training
    # if np.max(loss_list2) > max_ylim:
    #     max_ylim = np.max(loss_list2)

    # fig, ax = plt.subplots(1, 1, figsize=[7, 5])
    # ax.plot(loss_list1, lw=1, color = 'red', label = 'NMF')
    # ax.plot(loss_list2, lw=1, color = 'green', label = 'Classification model')
        
    # ax.set_title('Fit loss', fontsize=16)
    # ax.tick_params(labelsize=12)
    # ax.set_xlabel('Iteration', fontsize=14)
    # ax.set_ylabel('Loss', fontsize=14)
    # plt.ylim([0, max_ylim * 102/100])
    # plt.grid()
    # plt.legend(title="Loss")
    # plt.savefig('{}/Fit_loss.png'.format(output_path))
    # plt.show()

    # SNMF transforming
    tolerance_sample = tolerance / X_train.shape[0]

    # get X ~ H and weight of LR model
    X = summary['X'].copy()
    weight = summary['weight'].copy()

    # log
    sample_list = []
    loss_list_sample = []
    loss_list1_sample = []
    loss_list2_sample = []

    for X_now, y_now in [
        [X_train, y_train], [X_val, y_val]
    ]:
        # feature
        Y_pool = X_now.copy().values

        # label
        u_pool = y_now.copy().values

        for i in range(Y_pool.shape[0]):
            sample = sample_train['SampleID'][i]
            
            Y = Y_pool[[i]]
            u = u_pool[[i]]
            
            K, loss_list, loss_list1, loss_list2, best_i = SNMF_transform_sample(Y, X, iter, tolerance_sample, patience, epsStab, alpha, epsilon, u, weight)
            
            sample_list.append(sample)
            loss_list_sample.append(loss_list[best_i])
            loss_list1_sample.append(loss_list1[best_i])
            loss_list2_sample.append(loss_list2[best_i])

    # save summary
    summary['sample_list'] = sample_list
    summary['loss_list_sample'] = loss_list_sample
    summary['loss_list1_sample'] = loss_list1_sample
    summary['loss_list2_sample'] = loss_list2_sample

    # open a file in binary write mode
    with open('{}/summary.pkl'.format(output_path), 'wb') as file:
        # save the data to the file
        pickle.dump(summary, file)

if __name__ == '__main__':
    main()