from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        train_batch_size=12,
        unsupervised_batch_size=52,
        epochs=800,
        val_split_size=0.20,
        feature_size=64,
        alpha=0.7,
        patience=20,
        base_size=10,
        parallel=6,
        top_num=3,
        consistency_weight=0.5,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
