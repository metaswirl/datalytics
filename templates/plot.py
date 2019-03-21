import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import datalytics as da
import click

@click.command()
@click.option('--input', 'input_files', type=click.Path(), required=True)
@click.option('--output', 'output_file', type=click.Path(), required=True)
def main(input_file, output_file):
    data = da.read_df(input_file)
    f, ax = plt.subplots()

    f.savefig(output_file)

if __name__ == '__main__':
    main()
