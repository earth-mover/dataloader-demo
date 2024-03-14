# dataloader-demo

This repository demonstrates how you can build a performant cloud-native Pytorch dataloader using Zarr, Dask, Xarray, and Xbatcher.

## Demo

The demo can be run by executing the following command line script:

```shell
# cli help
python main.py --help

# example 1
python main.py \
    --batch-size 4 \
    --num-epochs 3 \
    --num-batches 500 \
    --shuffle \
    --source arraylake > logs-blog/fig2-log.txt

# example 2
python main.py \
    --batch-size 4 \
    --num-epochs 3 \
    --num-batches 500 \
    --num-workers 32 \
    --persistent-workers \
    --dask-threads 4 \
    --shuffle \
    --prefetch-factor 3 \
    --source arraylake > logs-blog/fig3-log.txt
```

The output of this script is a log file (`logs-blog/fig3-log.txt`) that can be visualized using the Jupyter Notebook (`plot.ipynb`).

## History

This work is started as a collaboration between Earthmover and [Zeus AI](https://myzeus.ai/). It leverages a number of open source projects, including Zarr, Dask, Xarray, and Xbatcher -- all of which have been supported by a number of grants from NSF, NASA, and CZI.

As of February 2024, some of the improvements discovered in this project are being [upstreamed into Xbatcher](https://github.com/xarray-contrib/xbatcher/pull/202).

### License

MIT
