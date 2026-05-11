import torch


def make_iid_partitions(num_samples, n_clients, shuffle=True, seed=None):
    """Create IID partitions of sample indices for n_clients."""
    idx = torch.arange(num_samples)
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        idx = idx[torch.randperm(num_samples, generator=generator)]
    splits = torch.tensor_split(idx, n_clients)
    return [split.tolist() for split in splits]


def make_label_shard_partitions(labels, n_clients, classes_per_client, seed=None):
    """Create label-shard non-IID partitions for n_clients.

    This is the label-sorted shard partition used in the referenced Keras code.
    """
    labels_tensor = torch.tensor(labels).flatten()
    n_shards = n_clients * classes_per_client
    if n_shards <= 0:
        raise ValueError('classes_per_client must be positive')

    idx_sorted = torch.argsort(labels_tensor)
    shards = torch.tensor_split(idx_sorted, n_shards)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    shard_order = torch.randperm(n_shards, generator=generator)

    partitions = []
    for client_id in range(n_clients):
        client_shards = shard_order[client_id * classes_per_client:(client_id + 1) * classes_per_client]
        client_idx = torch.cat([shards[s] for s in client_shards.tolist()])
        partitions.append(client_idx.tolist())
    return partitions


def classes_per_client(dataset_name, partition_level):
    """Return the number of label shards per client for a non-IID level."""
    if partition_level == 0:
        return None

    mapping = {
        'mnist':    {1: 6, 2: 4, 3: 2},
        'cifar10':  {1: 6, 2: 4, 3: 2},
        'cifar100': {1: 60, 2: 40, 3: 20},
    }
    dataset_mapping = mapping.get(dataset_name)
    if dataset_mapping is None:
        raise ValueError(f'Unsupported dataset for non-IID partitions: {dataset_name}')
    if partition_level not in dataset_mapping:
        raise ValueError(f'Unsupported partition level: {partition_level}')
    return dataset_mapping[partition_level]


def partition_train_indices(dataset_name, labels, n_clients, partition_level=0, seed=None):
    """Return train indices for each client according to partition_level."""
    if partition_level == 0:
        return make_iid_partitions(len(labels), n_clients, shuffle=True, seed=seed)

    classes = classes_per_client(dataset_name, partition_level)
    return make_label_shard_partitions(labels, n_clients, classes, seed=seed)
