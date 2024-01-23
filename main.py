import json
import multiprocessing
import time
from typing import Optional

import torch
import typer
import xbatcher
from arraylake import Client, config
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Annotated


def print_json(obj):
    print(json.dumps(obj))


class XBatcherPyTorchDataset(TorchDataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        t0 = time.time()
        print_json(
            {
                "event": "get-batch start",
                "time": t0,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
            }
        )
        # compute before stacking
        batch = self.bgen[idx].compute()

        # Use to_stacked_array to stack without broadcasting,
        # zeus' dataset.preprocess_inputs merges the "variable" and "level" dimensions
        # and likes this order: "time", "batch", "longitude", "latitude"
        # Hmm... This will be slow, since it constructs a multiindex.
        stacked = batch.to_stacked_array(
            new_dim="batch", sample_dims=("time", "longitude", "latitude")
        ).transpose("time", "batch", ...)
        x = torch.tensor(stacked.data)

        t1 = time.time()
        print_json(
            {
                "event": "get-batch end",
                "time": t1,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
                "duration": t1 - t0,
            }
        )
        return x


def setup():
    patch_size = 32
    input_steps = 3
    output_steps = 0
    jitter = 16

    client = Client()
    # set s3 endpoint for GCS + anonymous access
    config.set({"s3.endpoint_url": "https://storage.googleapis.com", "s3.anon": True})
    repo = client.get_repo("earthmover-public/weatherbench2")

    # opened with xarray lazy indexing, no dask
    ds = repo.to_xarray("datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative")

    DEFAULT_VARS = [
        "10m_wind_speed",
        "2m_temperature",
        "specific_humidity",  # Q(JH): this field has a a "level" dimension, how is that treated?
    ]

    ds = ds[DEFAULT_VARS]
    patch = dict(
        latitude=patch_size + jitter,
        longitude=patch_size + jitter,
        time=input_steps + output_steps,
    )
    overlap = dict(latitude=32, longitude=32, time=input_steps // 3 * 2)

    bgen = xbatcher.BatchGenerator(
        # dask chunk sizes aligning with patch size
        # This should allow parallel loading of multiple variables
        ds.chunk(patch),
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )

    dataset = XBatcherPyTorchDataset(bgen)

    return dataset


def main(
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 8,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 16,
    shuffle: Annotated[Optional[bool], typer.Option()] = None,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
):
    data_params = {
        "batch_size": batch_size,
    }
    if shuffle is not None:
        data_params["shuffle"] = True
    if num_workers is not None:
        data_params["num_workers"] = num_workers
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory

    run_start_time = time.time()
    print_json({"event": "run start", "time": run_start_time, "data_params": data_params})

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset = setup()
    training_generator = DataLoader(dataset, **data_params)
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    for i, sample in enumerate(training_generator):
        tt0 = time.time()
        print_json({"event": "training start", "batch": i, "time": tt0})
        time.sleep(0.3)  # simulate model training
        tt1 = time.time()
        print_json({"event": "training end", "batch": i, "time": tt1, "duration": tt1 - tt0})
        if i == num_batches:
            break

    run_finish_time = time.time()
    print_json(
        {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    )


if __name__ == "__main__":
    typer.run(main)
