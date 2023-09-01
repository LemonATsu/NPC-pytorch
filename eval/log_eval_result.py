import os
import cv2
import glob
import numpy as np
from copy import deepcopy

"""
Assuming directory structure as follows:
    [args.path] /
        [method_name 1] /
            [subject 1-N].npy
        [method_name 2] /
            [subject 1-N].npy
"""

def get_better_fn(better):
    if better == 'h':
        return np.maximum, -np.inf
    else:
        return np.minimum, np.inf

def metric_value_to_value_str(value, metric):
    if metric == 'psnr':
        return f'{value:.2f}'
    elif metric == 'ssim':
        return f'{value:.3f}'
    elif metric in ['lpips', 'lpips_alex']:
        return f'{value:.3f}'
    elif metric == 'kid':
        return f'{value*100:.2f}'
    elif metric == 'fid':
        return f'{value:.1f}'
    raise NotImplementedError

def print_table(
    metric_dict, 
    metric_to_print=[('psnr', 'h'), ('ssim', 'h')], 
    row_loc='l',
    col_loc='c',
    no_avg=False,
):
    """ Best values are along row
    metrics are separated by colums.
    metric_to_print: a list of tuples, each tuple is (metric_name, better). 
                     better indicates 'h'igher the better or 'l'ower the better
    """
    row_names = list(metric_dict.keys())
    col_names = list(metric_dict[row_names[0]].keys())
    num_metrics = len(metric_to_print)
    
    # find best values, and also create final average
    metric_best = {row_name: {k[0]: None for k in metric_to_print} 
                    for row_name in row_names} 
    # find the best one along each row.
    for row_name in row_names:
        row_dict = metric_dict[row_name]
        # first: find the best one for each metric
        for k, better in metric_to_print:
            best_fn, cur_best = get_better_fn(better)
            for col_name in col_names:
                col_dict = row_dict[col_name]
                try:
                    metric_val = np.array(col_dict[k]).mean()
                except:
                    import pdb; pdb.set_trace()
                    print

                cur_best = best_fn(cur_best, metric_val)
                if metric_val == cur_best:
                    metric_best[row_name][k] = col_name
                        
    # last row: average of all
    metric_best['Avg'] = {k[0]: None for k in metric_to_print}
    last_row_metric = {col_name: {k[0]: None for k in metric_to_print} for col_name in col_names}
    for col_name in col_names:
        for k, better in metric_to_print:
            all_row_values = []
            for row_name in row_names:
                all_row_values.append(
                    metric_dict[row_name][col_name][k]
                )
            avg_values = np.concatenate(all_row_values).mean()
            last_row_metric[col_name][k] = avg_values
    for k, better in metric_to_print:
        best_fn, cur_best = get_better_fn(better)
        for col_name in col_names:
            metric_val = last_row_metric[col_name][k]
            cur_best = best_fn(cur_best, metric_val)
            if metric_val == cur_best:
                metric_best['Avg'][k] = col_name
    # now, let's create the string, one by one.
    # header
    header = '\\begin{tabular}{' + (row_loc + col_loc * num_metrics * len(col_names)) + '}'

    # # column names
    column_header = ''
    for col_name in col_names:
        column_header += '& \\multicolumn{' + str(num_metrics) + '}{c}{' + col_name.replace('_', '-') + '} '
    column_header += '\\\\'

    metric_header = ''
    for col_name in col_names:
        for metric_name, better in metric_to_print:
            if better == 'h':
                symbol = '$\\uparrow$'
            else:
                symbol = '$\\downarrow$'
            metric_header += ' & ' + metric_name.upper().replace('KID', 'KIDx100') + '~' + symbol + ' '
    metric_header += '\\\\'
            
    row_strings = []

    for row_name in row_names:
        row_dict = metric_dict[row_name]
        row_string = row_name.replace('_', '-') + ' '
        for col_name in col_names:
            col_dict = row_dict[col_name]
            for metric_name, better in metric_to_print:
                metric_val = np.array(col_dict[metric_name]).mean()
                val_str = metric_value_to_value_str(metric_val, metric_name)
                
                if metric_best[row_name][metric_name] == col_name:
                    row_string += f'& \\textbf{{{val_str}}}'
                else:
                    row_string += f'& {val_str}'
        row_strings.append(row_string)

    # last row
    row_strings.append('\\midrule')
    row_string = 'Avg' + ' '
    for col_name in col_names:
        for metric_name, better in metric_to_print:
            metric_val = last_row_metric[col_name][metric_name]
            val_str = metric_value_to_value_str(metric_val, metric_name)
            if metric_best['Avg'][metric_name] == col_name:
                row_string += f'& \\textbf{{{val_str}}}'
            else:
                row_string += f'& {val_str}' 
    row_strings.append(row_string)


    # now: create cmidrule to group metrics
    cmidrule_string = ''
    for i in range(len(col_names)):
        col_start = 2 + i * num_metrics
        col_end = 2 + (i+1) * num_metrics - 1
        midrule_range = f'{col_start}-{col_end}'
        cmidrule_string += f'\\cmidrule(lr){{{midrule_range}}}'
    cmidrule_string += '%'

    print(f'row_names {row_names}')
    print(f'col_names {col_names}')
    print(header)
    print('\\toprule')
    print(column_header)
    print(cmidrule_string)
    print(metric_header)
    print('\\midrule')
    for i, row_string in enumerate(row_strings):
        if i % 2 == 1:
            print('\\rowcolor{Gray}')
        if row_string != '\\midrule':
            print(row_string + '\\\\')
        else:
            print(row_string)
    print('\\end{tabular}')

def print_table_col(
    metric_dict, 
    metric_to_print=[('psnr', 'h'), ('ssim', 'h')], 
    row_loc='l',
    col_loc='c',
    no_avg=False,
):
    """ Best values along column
    """
    # TODO: so far it's all identical?
    col_names = list(metric_dict.keys())
    row_names = list(metric_dict[col_names[0]].keys())
    num_metrics = len(metric_to_print)
    
    # find best values, and also create final average
    metric_best = {col_name: {k[0]: None for k in metric_to_print} 
                    for col_name in col_names} 
    

                        
    metric_best['Avg'] = {k[0]: None for k in metric_to_print}
    last_col_metric = {row_name: {k[0]: None for k in metric_to_print} for row_name in row_names}
    for row_name in row_names:
        for k, better in metric_to_print:
            all_col_values = []
            for col_name in col_names:
                try:
                    all_col_values.append(
                    metric_dict[col_name][row_name][k]
                )
                except:
                    import pdb; pdb.set_trace()
                    print
            avg_values = np.concatenate(all_col_values).mean()
            last_col_metric[row_name][k] = avg_values
    n_cols = len(col_names)
    if not no_avg:
        n_cols += 1
    header = '\\begin{tabular}{' + (row_loc + col_loc * num_metrics * (len(col_names) + 1)) + '}'

    if not no_avg:
        metric_dict['Avg'] = last_col_metric


    # find the best one along each col
    col_names = list(metric_dict.keys())
    for col_name in col_names:
        col_dict = metric_dict[col_name]
        for k, better in metric_to_print:
            best_fn, cur_best = get_better_fn(better)
            for row_name in row_names:
                row_dict = col_dict[row_name]
                try:
                    metric_val = np.array(row_dict[k]).mean()
                except:
                    import pdb; pdb.set_trace()
                    print

                cur_best = best_fn(cur_best, metric_val)
                if metric_val == cur_best:
                    metric_best[col_name][k] = row_name
    # # column names
    column_header = ''
    for col_name in col_names:
        column_header += '& \\multicolumn{' + str(num_metrics) + '}{c}{' + col_name.replace('_', '-') + '} '
    column_header += '\\\\'

    metric_header = ''
    for col_name in col_names:
        for metric_name, better in metric_to_print:
            if better == 'h':
                symbol = '$\\uparrow$'
            else:
                symbol = '$\\downarrow$'
            metric_header += ' & ' + metric_name.upper().replace('KID', 'KIDx100') + '~' + symbol + ' '
    metric_header += '\\\\'
            
    row_strings = []

    # create row string
    row_strings = []
    """
    for col_name in col_names:
        col_dict = metric_dict[col_name]
        row_string = col_name
        for row_name in row_names:
            row_dict = col_dict[row_name]
            for metric_name, better in metric_to_print:
                metric_val = np.array(row_dict[metric_name]).mean()
                val_str = metric_value_to_value_str(metric_val, metric_name)
                
                if metric_best[col_name][metric_name] == row_name:
                    row_string += f'& \\textbf{{{val_str}}}'
                else:
                    row_string += f'& {val_str}'
        row_strings.append(row_string)
    """
    for row_name in row_names:
        row_string = row_name

        for col_name in col_names:
            col_dict = metric_dict[col_name][row_name]
            for metric_name, better in metric_to_print:
                metric_val = np.array(col_dict[metric_name]).mean()
                val_str = metric_value_to_value_str(metric_val, metric_name)
                if metric_best[col_name][metric_name] == row_name:
                    row_string += f'& \\textbf{{{val_str}}}'
                else:
                    row_string += f'& {val_str}'
        row_strings.append(row_string)


    # now: create cmidrule to group metrics
    cmidrule_string = ''
    for i in range(len(col_names)):
        col_start = 2 + i * num_metrics
        col_end = 2 + (i+1) * num_metrics - 1
        midrule_range = f'{col_start}-{col_end}'
        cmidrule_string += f'\\cmidrule(lr){{{midrule_range}}}'
    cmidrule_string += '%'
    print(header)
    print('\\toprule')
    print(column_header)
    print(cmidrule_string)
    print(metric_header)
    print('\\midrule')
    for i, row_string in enumerate(row_strings):
        if i % 2 == 1:
            print('\\rowcolor{Gray}')
        if row_string != '\\midrule':
            print(row_string + '\\\\')
        else:
            print(row_string)
    print('\\end{tabular}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument('-p', '--path', default='eval_results/h36m_zju/', 
                        type=str, help='path to the evaluation results')
    parser.add_argument('-b', '--baselines', 
                        default='A-NeRF,DANBO,NeuralBody,Anim-NeRF,ARAH,TAVA', 
                        type=str,
                        help='comma separated list of baseline methods to log')
    parser.add_argument('-o', '--ours',
                        default='Ours-lbs-only,Ours-full',
                        type=str,
                        help='comma separated list of our methods to log')
    parser.add_argument('-s', '--subset', default='novel_pose', type=str,
                        help='subset of the dataset to log')
    parser.add_argument('-c', '--col', action='store_true',
                        help='print table in column format')
    parser.add_argument('-n', '--no-avg', action='store_true',
                        help='do not print average row')
    args = parser.parse_args()

    path = args.path
    subset = args.subset
    baselines = args.baselines.split(',')
    ours = args.ours.split(',')

    baseline_dirs = []
    for b in baselines:
        p = os.path.join(path, b)
        if os.path.exists(p):
            baseline_dirs.append(p)

    ours_dirs = []
    for o in ours:
        p = os.path.join(path, o)
        if os.path.exists(p):
            ours_dirs.append(p)
    
    #subjects = [os.path.basename(p).split('.')[0]
    #            for p in sorted(glob.glob(os.path.join(ours_dirs[-1], '*.npy')))]

    if 'h36m' in path:
        #subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        subjects = ['S9']
    elif 'perfcap' in path:
        #subjects = ['nadia', 'weipeng']
        subjects = ['weipeng']
    elif 'animal' in path:
        subjects = ['wolf']
    else:
        raise NotImplementedError
    print(subjects)

    method_dirs = baseline_dirs + ours_dirs
    method_names = baselines + ours
    print(method_dirs)
    print(method_names)
    result_dicts = {s: {m: {} for m in method_names} for s in subjects}

    for i, (method_dir, method_name) in enumerate(zip(method_dirs, method_names)):

        for subject in subjects:
            results = np.load(
                        os.path.join(method_dir, f'{subject}.npy'), 
                        allow_pickle=True
                    ).item()

            result_dicts[subject][method_name] = results[method_name][subset]
            if f'{subset}_cropped' in results[method_name]:
                result_dicts[subject][method_name].update(results[method_name][f'{subset}_cropped'])
    print_fn = print_table_col if args.col else print_table

    print_fn(
        result_dicts,
        #metric_to_print=[('psnr', 'h'), ('ssim', 'h'), ('lpips', 'l')], 
        #metric_to_print=[('psnr', 'h'), ('ssim', 'h')], 
        #metric_to_print=[('psnr', 'h'), ('ssim', 'h'), ('fid', 'l'), ('kid', 'l'), ('lpips', 'l')], 
        metric_to_print=[('kid', 'l'), ('lpips', 'l'), ('lpips_alex', 'l')], 
        row_loc='l',
        col_loc='c',
        no_avg=args.no_avg,
    )


