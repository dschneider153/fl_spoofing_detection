import os
import io
import csv
import pandas as pd
import zstandard as zstd # type: ignore

def decompress_zst_to_csv(zst_file_path, output_csv_path):
    dctx = zstd.ZstdDecompressor()
    data_list = []

    with open(zst_file_path, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            csv_reader = csv.reader(text_stream)
            for row in csv_reader:
                data_list.append(row)

    df = pd.DataFrame(data_list)
    df.to_csv(output_csv_path, index=False)

zst_file = 'data/zst/xnas-itch-20260130.trades.csv.zst'
os_output_path = os.path.join('data', 'csv', 'output.csv')

decompress_zst_to_csv(zst_file, os_output_path)

