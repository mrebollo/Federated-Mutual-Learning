import argparse


def combine(start_id=1, output='config.txt'):
    dataset_models = {
        'mnist': ['MLP', 'LeNet5'],
        'femnist': ['MLP', 'LeNet5'],
        'cifar10': ['CNN1', 'CNN2'],
        'cifar100': ['CNN1', 'CNN2'],
    }
    partitions = [0, 1, 2, 3]
    algorithms = ['fed_avg', 'fed_prox', 'fed_mutual', 'normal']

    with open(output, 'w') as f:
        f.write('ArrayTaskID\tdataset\tpartition\tmodel\talgorithm\n')
        idx = start_id
        for dataset, models in dataset_models.items():
            for model in models:
                for partition in partitions:
                    for algorithm in algorithms:
                        f.write(f'{idx}\t{dataset}\t{partition}\t{model}\t{algorithm}\n')
                        idx += 1

    print(f'Generated {output} with {idx - start_id} experiments starting at {start_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a config.txt for slurm array experiments.')
    parser.add_argument('--start', type=int, default=1,
                        help='Initial experiment ID')
    parser.add_argument('--output', type=str, default='config.txt',
                        help='Output config filename')
    args = parser.parse_args()
    combine(start_id=args.start, output=args.output)
