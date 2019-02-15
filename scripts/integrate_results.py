"""Integrate manual counts with automated measures of cell and nuclear volumes."""

import re

from collections import defaultdict

import click

import pandas as pd


def simple_slugify(input_string):

    return re.sub('[ #]', '_', input_string)


def integrate_counts(manual_counts_fpath, cell_data_fpath, nuclei_data_fpath):

    manual_counts_df = pd.read_csv(manual_counts_fpath)

    def parse_raw_file(fpath):

        data = defaultdict(dict)
        raw_lines = [l.strip().split(',') for l in open(fpath).readlines()]
    
        for name, series, val in raw_lines:
            data[name][series] = float(val)

        return data

    cell_data = parse_raw_file(cell_data_fpath)
    nuclei_data = parse_raw_file(nuclei_data_fpath)

    def parse_name(name):
        fname, series, _ = name.split('_')
        return simple_slugify(fname), series
    
    parsed_names = list(parse_name(name) for name in manual_counts_df['Name'])

    filenames, series = zip(*parsed_names)
    # import pdb; pdb.set_trace()
    cell_volumes = [cell_data[n][s] for n, s in parsed_names]
    nuclear_volumes = [nuclei_data[n][s] for n, s in parsed_names]
    nuclear_counts = manual_counts_df['Nuclei_per_Cell']
    avg_nuclear_volumes = [float(nv) / int(nc) for nv, nc in zip(nuclear_volumes, nuclear_counts)]

    integrated_results = pd.DataFrame(
        {'filename': filenames,
         'series': series,
         'cell_volume': cell_volumes,
         'total_nuclear_volume': nuclear_volumes,
         'nuclei_per_cell': nuclear_counts,
         'avg_nuclear_vol': avg_nuclear_volumes
        }
    )
    integrated_results.round(2)
    integrated_results.to_csv('results/integrated.csv', float_format='%.2f')


@click.command()
@click.argument('manual_counts_fpath')
@click.argument('raw_cell_data')
@click.argument('raw_nuclei_data')
def main(manual_counts_fpath, raw_cell_data, raw_nuclei_data):

    integrate_counts(manual_counts_fpath, raw_cell_data, raw_nuclei_data)


if __name__ == "__main__":
    main()
