from collections import defaultdict
from typing import Any

import pandas as pd


class CsvReport:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.raw_dict: dict[str, Any] = defaultdict(list)

    def add_record(self, **kwargs) -> None:  # pylint: disable=unused-argument
        for k, v in kwargs.items():
            self.raw_dict[k].append(v)

    def add_columns(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.raw_dict[k] = v

    def dump(self, **kwargs) -> None:  # pylint: disable=unused-argument
        # Ensure all the columns in raw_dict have the same length
        col_lens = [len(v) for v in self.raw_dict.values()]
        if len(set(col_lens)) != 1:
            raise ValueError(
                f"Columns in raw_dict have different lengths: "
                f"{col_lens} for {list(self.raw_dict.keys())}"
            )

        df = pd.DataFrame(self.raw_dict)
        df.to_csv(self.csv_path, index=False)
