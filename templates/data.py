import pandas as pd
import datalytics as da
import click

@click.command()
@click.argument('input_fpath', type=click.Path())
@click.argument('output_fpath', type=click.Path())
def main(input_fpath, output_fpath):
    da.read_f(input_fpath)


    result = pd.DataFrame()
    da.write_df(result, output_fpath)

if __name__ == '__main__':
    main()
