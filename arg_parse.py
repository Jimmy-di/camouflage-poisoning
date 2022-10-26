"""Implement an ArgParser common to both brew_poison.py"""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')


    ###########################################################################
    # Training params:
    parser.add_argument('--model_name', default='ResNet18', type=str, choices=['ResNet18', 'ResNet34', 'ResNet50', 'VGG16', 'VGG11', 'MLP', 'MobileNet'])
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR-Binary', 'ImageNet', 'Imagenette', 'Imagewoof'])
    parser.add_argument('--epochs', default=40, type=int, help='Training Epochs')
    parser.add_argument('--batchsize', default=128, type=int, help='Training batch size during optimization')
    parser.add_argument('--eta', default=0.01, type=float, help='Initial LR for training')
    # Reproducibility management:
    parser.add_argument('--seed', default=None, type=int, help='Initialize the setup with this key.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--pbudget', default=0.01, type=float, help='Fraction of training data that is used as poisons')
    parser.add_argument('--cbudget', default=0.01, type=float, help='Fraction of training data that is used as camouflages')
    parser.add_argument('--targets', default=1, type=int, help='Number of targets')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--log_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--camou_path', default='camouflage/', type=str)
    parser.add_argument('--data_path', default='~/data/', type=str)
    parser.add_argument('--model_path', default='model/', type=str)

    ###########################################################################



    # Poison brewing:
    parser.add_argument('--attackoptim', default='Adam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')

    # Optimization setup
    parser.add_argument('--savemodel', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNette and Imagewoof].')
    parser.add_argument('--loadmodel', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNette and Imagewoof].')
    parser.add_argument('--save', default=None, help='Save results in full / ')


    return parser
