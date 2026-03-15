import json
from torch.utils.data import Dataset, DataLoader

class VideoQuestionDataset(Dataset):
    def __init__(self, json_file):
        # 读取 JSON 文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据
        item = self.data[idx]
        instruction = item['instruction']
        video_id = item['visual_input']
        output = item['output']
        task = item['task']
        
        return instruction, video_id, output, task

def collate_fn(batch):
    # 将 batch 转换为字典
    instructions, video_ids, outputs, tasks = zip(*batch)
    return {
        'instruction': instructions,
        'video_id': video_ids,
        'output': outputs,
        'task': tasks
    }

# 创建数据集和 DataLoader
# json_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/fesvhtr/FunQA/main/FunQA_test.json'  # 替换为你的 JSON 文件路径
# dataset = VideoQuestionDataset(json_file)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
# for batch in dataloader:
#     print(batch['instruction'][0])
#     print(batch['video_id'])
#     print(batch['output'])
#     break