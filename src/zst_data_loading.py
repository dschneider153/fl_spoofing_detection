import io
import os

import pandas as pd
import zstandard as zstd  # type: ignore


# Decompress function
def decompress_zst_to_csv(zst_file_path, output_csv_path):
    with open(zst_file_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            df = pd.read_csv(text_stream)
    df.to_csv(output_csv_path, index=False)
    print(f"Done: {output_csv_path}, rows: {len(df)}")


# Path definitions
zst_file = "data/MBP-10/zst/combinedmbp-10.csv.zst"
os_output_path = os.path.join("data", "MBP-10", "csv", "output.csv")

decompress_zst_to_csv(zst_file, os_output_path)
