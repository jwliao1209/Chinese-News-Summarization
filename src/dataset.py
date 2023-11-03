from torch.utils.data.dataset import Dataset


class ChineseNewsDataset(Dataset):
    def __init__(self, data_list, transform=False):
        self.data_list = [transform(data) for data in tqdm(data_list)] \
                         if transform is not None else data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {k: v if k in [SUMMARY_COL] else torch.tensor(v) for k, v in data_list_dict.items()}
    return data_tensor_dict
