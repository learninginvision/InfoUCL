import os
import torch
import numpy as np
from tqdm import tqdm
from utils.arguments import get_args
from models import get_model
from datasets import get_dataset
from metrics import knn_monitor, save_results
from tensorboardX import SummaryWriter

def main(device, args):
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, _, _ = dataset_copy.get_data_loaders(args)
    knn_acc_results = []

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    writer = SummaryWriter(os.path.join(args.log_dir, 'summary')) # tensorboard
    
    # get data loaders for each task
    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(dataset.N_TASKS):
        tr, me, te = dataset.get_data_loaders(args)
        train_loaders.append(tr)
        memory_loaders.append(me)
        test_loaders.append(te)
    
    it_count = 0
    eval_tids = [j for j in range(dataset.N_TASKS)]
    for t in range(dataset.N_TASKS):

        # train model
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
        for epoch in global_progress:
            model.train()
            
            local_progress=tqdm(train_loaders[t], desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                data_dict = model.observe(images1, labels, images2, notaug_images)

            # write to tensorboard
            for k, v in data_dict.items():
                data_dict[k] = v.mean().item() if isinstance(v, torch.Tensor) else v
                writer.add_scalar(f'{k}', data_dict[k], it_count)
            it_count += 1
            global_progress.set_postfix(data_dict)

        # save model
        model_path = os.path.join(args.log_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
        torch.save({
            'epoch': 200,
            'state_dict':model.net.state_dict()
            }, model_path)
        
        # test current model on all tasks
        knn_acc_list = []
        for i in eval_tids:
            acc, _ = knn_monitor(model.net.module.backbone, dataset, memory_loaders[i], test_loaders[i],
                                 device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
            knn_acc_list.append(acc)

        # memorize acc of current model on all tasks
        knn_acc_results.append(knn_acc_list)

        # process end of task
        if hasattr(model, 'end_task'):
            model.end_task(dataset)

    return knn_acc_results

    
if __name__ == "__main__":
    
    # get arguments
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    
    # run main
    results = main(device=args.device, args=args)
    
    # save results
    save_results(results, save_path=args.log_dir)

    # rename log dir to completed
    completed_log_dir = args.log_dir.replace('in-progress', 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')