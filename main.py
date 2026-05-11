import torch
from tqdm import tqdm
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from Results import ResultsWriter
from utils import LR_scheduler, Recorder, Catfish, Summary
from Trainer import Trainer

def main():
    # init args
    args = args_parser()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Running on', args.device)
    data = Data(args)
    args.dataset_size = len(data.trainset)
    Train = Trainer(args)

    # init nodes
    Global_node = Global_Node(data.test_loader, args)
    Node_List = [Node(k, data.train_loader[k], data.test_loader, args) for k in range(args.node_num)]
    Catfish(Node_List, args)

    # init variables
    recorder = Recorder(args)
    results = ResultsWriter(args)
    Summary(args)
    # start
    for rounds in tqdm(range(args.R), desc='Rounds'):
        LR_scheduler(rounds, Node_List, args)
        for k, node in enumerate(Node_List):
            node.fork(Global_node)
            for epoch in range(args.E):
                Train(node)
            node_loss, node_acc = recorder.validate(node, use_meme=(args.algorithm != 'normal'))
            results.write_agent_round(k, rounds + 1, node_loss, node_acc)
        Global_node.merge(Node_List)
        recorder.validate(Global_node)
        global_loss = recorder.val_loss['0'][-1]
        if hasattr(global_loss, 'item'):
            global_loss = global_loss.item()
        global_acc = recorder.val_acc['0'][-1]
        print(f"Round {rounds + 1}: global_loss={global_loss:.4f} global_acc={global_acc:.2f}%")
    recorder.finish()
    results.finish()
    Summary(args)


if __name__ == '__main__':
    main()

