import torch
from tqdm import tqdm
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Catfish, Summary
from Trainer import Trainer

def main():
    # init args
    args = args_parser()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Running on', args.device)
    data = Data(args)
    Train = Trainer(args)

    # init nodes
    Global_node = Global_Node(data.test_all, args)
    Node_List = [Node(k, data.train_loader[k], data.test_loader, args) for k in range(args.node_num)]
    Catfish(Node_List, args)

    # init variables
    recorder = Recorder(args)
    Summary(args)
    # start
    for rounds in tqdm(range(args.R), desc='Rounds'):
        LR_scheduler(rounds, Node_List, args)
        for k in range(len(Node_List)):
            Node_List[k].fork(Global_node)
            for epoch in range(args.E):
                Train(Node_List[k])
                # recorder.validate(Node_List[k])
        Global_node.merge(Node_List)
        recorder.validate(Global_node)
        global_loss = recorder.val_loss['0'][-1]
        if hasattr(global_loss, 'item'):
            global_loss = global_loss.item()
        global_acc = recorder.val_acc['0'][-1]
        print(f"Round {rounds + 1}: global_loss={global_loss:.4f} global_acc={global_acc:.2f}%")
    recorder.finish()
    Summary(args)


if __name__ == '__main__':
    main()
