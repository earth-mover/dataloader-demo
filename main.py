import json
import time
from typing import Optional

import dask
import torch
import typer
import xbatcher
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch import multiprocessing
from typing_extensions import Annotated

from dask.cache import Cache

# torch.multiprocessing.set_sharing_strategy('file_descriptor')
fast = {}
# slow = zict.Func(pickle.dumps, pickle.loads, zict.File('storage/'))
# def weight(k, v):
#     return sys.getsizeof(v)
# cache_data = zict.Buffer(fast, slow, 1e9, weight=weight) # 1gb in-memory cache
cache = Cache(1e10, cache_data=fast)  # 10gb cache, split between in-memory and file storage
cache.register()


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
        # load before stacking
        batch = self.bgen[idx].load()

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


def setup(patch_size: int = 32, input_steps: int = 3, output_steps: int = 0, jitter: int = 16):
    # client = Client()
    # # set s3 endpoint for GCS + anonymous access
    # config.set({"s3.endpoint_url": "https://storage.googleapis.com", "s3.anon": True})
    # repo = client.get_repo("earthmover-public/weatherbench2")

    # opened with dask
    # ds = repo.to_xarray("datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative")
    ds = xr.open_dataset(
        "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr",
        engine="zarr",
        chunks={},
    )

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
        ds,
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )

    dataset = XBatcherPyTorchDataset(bgen)

    return dataset


def main(
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 16,
    shuffle: Annotated[Optional[bool], typer.Option()] = None,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    multiprocessing_context: Annotated[Optional[str], typer.Option()] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    train_step_time: Annotated[Optional[float], typer.Option()] = 0.1,
    dask_threads: Annotated[Optional[int], typer.Option()] = None,
    dask_cache: Annotated[Optional[int], typer.Option(min=0, max=1e11)] = None,
):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    data_params = {
        "batch_size": batch_size,
    }
    if shuffle is not None:
        data_params["shuffle"] = shuffle
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        if multiprocessing_context is None:
            data_params["multiprocessing_context"] = "forkserver"
    if multiprocessing_context is not None:
        if multiprocessing_context == "loky":
            from joblib.externals.loky.backend.context import get_context

            data_params["multiprocessing_context"] = get_context("loky")
        else:
            data_params["multiprocessing_context"] = multiprocessing_context
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)
    if dask_cache is not None:
        # cache = Cache(dask_cache)
        # cache.register()
        raise ValueError("remember to set this globally")

    run_start_time = time.time()
    print_json(
        {
            "event": "run start",
            "time": run_start_time,
            "data_params": str(data_params),
            "locals": _locals,
        }
    )

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset = setup()
    training_generator = DataLoader(dataset, **data_params)
    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    for epoch in range(num_epochs):
        e0 = time.time()
        print_json({"event": "epoch start", "epoch": epoch, "time": e0})

        for i, sample in enumerate(training_generator):
            tt0 = time.time()
            print_json({"event": "training start", "batch": i, "time": tt0})
            time.sleep(train_step_time)  # simulate model training
            tt1 = time.time()
            print_json({"event": "training end", "batch": i, "time": tt1, "duration": tt1 - tt0})
            if i == num_batches - 1:
                break

        e1 = time.time()
        print_json({"event": "epoch end", "epoch": epoch, "time": e1, "duration": e1 - e0})

    run_finish_time = time.time()
    print_json(
        {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    )


if __name__ == "__main__":
    typer.run(main)
