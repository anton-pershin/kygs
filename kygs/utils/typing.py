from typing import Literal

import numpy as np
from numpy.typing import NDArray


NDArrayInt = NDArray[np.int32 | np.int64]
NDArrayFloat = NDArray[np.float64 | np.float32]

TimeUnit = Literal["year", "month", "day", "hour", "minute", "second"]

