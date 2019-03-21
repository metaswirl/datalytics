import pandas as pd
import datalytics as da
import click

@click.command()
@click.option('--input', 'input_file', type=click.Path(), required=True)
@click.option('--output', 'output_file', type=click.Path(), required=True)
def main(input_file, output_file):
    data = da.read_df(input_file)
    result = pd.DataFrame()
    da.write_df(result, output_file)

if __name__ == "__main__":
    main()
