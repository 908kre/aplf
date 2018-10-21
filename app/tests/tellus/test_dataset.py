from aplf.tellus.dataset import load_dataset_df, get_row, TellusDataset
import pandas as pd


def test_get_row():
    rows = get_row(
        base_path='/store/tellus/train',
        sat="LANDSAT",
        label_dir="positive",
        label=True
    )
    assert len(rows) == 1530



def test_dataset():
    output = load_dataset_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=True,
    )
    assert len(dataset[0]) == 4
