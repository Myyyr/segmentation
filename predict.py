import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model
import torch

import os

# 0 -> quadro
# 3 -> GTX2 


def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

def train(arguments, data_splits, n_split = 0):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug
    predict_path = arguments.predict_path

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model, im_dim = train_opts.im_dim, split=n_split)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    test_dataset  = ds_class(ds_path, split='test',  data_splits = data_splits['test'],  im_dim=train_opts.im_dim, transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=2, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    
        
        

    # Validation and Testing Iterations
    loader, split = [test_loader, 'test']
    for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

        # Make a forward pass with the model
        model.set_input(images, labels)
        model.predict(predict_path)

        # Error visualisation
        errors = model.get_current_errors()
        stats = model.get_segmentation_stats()
        error_logger.update({**errors, **stats}, split=split)

        # Visualise predictions
        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, epoch=1, save_result=False)

        del images, labels

    # Update the plots
    for split in ['test']:
        visualizer.plot_current_errors(1, error_logger.get_errors(split), split_name=split)
        visualizer.print_current_errors(1, error_logger.get_errors(split), split_name=split)
    error_logger.reset()
    # print("Memory Usage :", convert_bytes(torch.cuda.max_memory_allocated()))
    print("Number of parameters :", model.get_number_parameters())

    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('-g', '--gpu',  help='gpu to use', required=True)
    parser.add_argument('-p', '--predict_path',  help='prediction saving path', required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data_splits = {'train':[], 'test':[]}
    all_splits = ['split_'+str(i+1) for i in range(6)]

    i = 1
    
    data_splits['test'] = [all_splits[i]]
    data_splits['train'] = all_splits[:i] + all_splits[i+1:]

    train(args, data_splits, i+1)
