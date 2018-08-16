from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
from sklearn.model_selection import train_test_split
from dask import delayed
from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from .predict import predict


class Graph(object):
    def __init__(self,
                 dataset_dir,
                 output_dir,
                 epochs,
                 batch_size,
                 val_split_size,
                 patience,
                 parallel,
                 ):

        ids = list(range(parallel))

        dataset_df = delayed(load_dataset_df)(dataset_dir, 'train.csv')
        spliteds = pipe(
            ids,
            map(lambda x: delayed(train_test_split)(
                dataset_df, test_size=val_split_size, shuffle=True
            )),
            list
        )
        train_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[0])),
            map(delayed(TgsSaltDataset)),
            list
        )

        val_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[1])),
            map(delayed(TgsSaltDataset)),
            list
        )

        traineds = pipe(
            zip(train_datasets, val_datasets),
            enumerate,
            map(lambda x: delayed(train)(
                model_id=x[0],
                model_path=f"{output_dir}/model_{x[0]}.pt",
                train_dataset=x[1][0],
                val_dataset=x[1][1],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )),
            list
        )

        model_paths = pipe(
            traineds,
            map(delayed(lambda x: x["model_path"])),
            list
        )

        eval_train = delayed(predict)(
            model_paths=model_paths,
            output_dir=f"{output_dir}/train",
            dataset=train_datasets[0]
        )

        eval_val = delayed(predict)(
            model_paths=model_paths,
            output_dir=f"{output_dir}/val",
            dataset=val_datasets[0]
        )

        progresses = pipe(
            traineds,
            map(delayed(lambda x: x["progress"])),
            list
        )

        progress_file = pipe(
            progresses,
            enumerate,
            map(delayed(lambda x: x[1].to_json(f"{output_dir}/progress_{x[0]}.json"))),
            list
        )

        submission_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        submission_dataset = delayed(TgsSaltDataset)(
            submission_df,
            is_train=False
        )

        submission_df = delayed(predict)(
            model_paths=model_paths,
            output_dir=f"{output_dir}/sub",
            dataset=submission_dataset
        )
        submission_file = delayed(lambda df: df.to_csv(f"{output_dir}/submission.csv"))(submission_df)

        self.output = delayed(lambda x: x)((
            submission_df,
            submission_file
        ))
