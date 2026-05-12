import csv
from pathlib import Path
from datetime import datetime


DEFAULT_RESULTS_HEADER = [
    'dataset', 'model', 'net', 'size', 'iid', 'alg',
    'rep', 'round', 'agent', 'acc', 'loss', 'err', 'nmsg', 'msiz'
]


class ResultsWriter:
    def __init__(self, args, filename=None):
        self.args = args
        self.filename = Path(filename or 'results.csv')
        self.fieldnames = DEFAULT_RESULTS_HEADER
        self._ensure_file()

    def _ensure_file(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        if not self.filename.exists():
            with self.filename.open('w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def write_round(self, round_idx, global_loss, global_acc):
        row = self._make_agent_row(0, round_idx, global_loss, global_acc)
        with self.filename.open('a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row)

    def write_agent_round(self, agent_id, round_idx, global_loss, global_acc):
        row = self._make_agent_row(agent_id, round_idx, global_loss, global_acc)
        with self.filename.open('a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row)

    def _make_agent_row(self, agent_id, round_idx, loss, acc):
        args = self.args
        base_row = {
            'dataset': getattr(args, 'dataset', ''),
            'model': getattr(args, 'local_model', ''),
            'net': 'none',
            'size': getattr(args, 'node_num', ''),
            'iid': getattr(args, 'iid', ''),
            'alg': getattr(args, 'algorithm', ''),
            'rep': getattr(args, 'rep', 1),
            'round': round_idx,
            'agent': agent_id,
            'acc': float(acc / 100.0) if acc is not None else '',
            'loss': float(loss) if loss is not None else '',
            'err': 0,
            'nmsg': 0,
            'msiz': 0
        }
        return {field: base_row.get(field, '') for field in self.fieldnames}

    def finish(self):
        print(f'Results appended to {self.filename.resolve()}')
