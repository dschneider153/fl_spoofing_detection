import os
import io
import csv
import pandas as pd
import zstandard as zstd # type: ignore

def decompress_zst_to_csv(zst_file_path, output_csv_path):
    with open(zst_file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            df = pd.read_csv(text_stream)
    df.to_csv(output_csv_path, index=False)
    print(f"Done: {output_csv_path}, rows: {len(df)}")

zst_file = 'data/NVIDIA_TEST/xnas-itch-20260130.mbp-10.csv.zst'
os_output_path = os.path.join('data', 'NVIDIA_TEST', 'nvidia_mbp-10.csv')

decompress_zst_to_csv(zst_file, os_output_path)

