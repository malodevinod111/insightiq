import re

import pandas as pd

MAX_CHART_ROWS = 50
MAX_CHART_SERIES = 3
_TIME_NAME = re.compile(r"date|time|day|week|month|quarter|year", re.IGNORECASE)


def build_chart(table: pd.DataFrame | None) -> dict | None:
    """Pick a chart for a result table, or None when a chart wouldn't help.

    Deterministic on purpose: a label column plus numeric columns → bar chart,
    or line chart when the label looks like a date/period.
    """
    if table is None or not (2 <= len(table) <= MAX_CHART_ROWS):
        return None

    numeric = list(table.select_dtypes(include="number").columns)
    labels = [c for c in table.columns if c not in numeric]
    if not numeric or not labels:
        return None

    label = labels[0]
    if table[label].nunique() != len(table):
        return None  # repeated labels would be silently merged in the chart

    is_time = pd.api.types.is_datetime64_any_dtype(table[label]) or bool(_TIME_NAME.search(str(label)))
    data = table.set_index(label)[numeric[:MAX_CHART_SERIES]]
    if is_time:
        data = data.sort_index()
    return {"kind": "line" if is_time else "bar", "data": data}
