import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datalytics as da
import click

@click.command()
@click.argument('input_fpath', type=click.Path())
@click.argument('output_fpath', type=click.Path())
def main(input_fpath, output_fpath):
    df = da.read_df(input_fpath)

    f, ax = plt.subplots(1,1)

    f.savefig(output_fpath)

if __name__ == '__main__':
    main()
