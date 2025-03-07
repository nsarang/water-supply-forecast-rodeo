import uuid

import dask_expr as dx


def _dx_cross_merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    suffixes=("_x", "_y"),
    indicator=False,
    npartitions=None,
    # shuffle=None,
    broadcast=None,
):
    """
    See merge.__doc__ with how='cross'
    """

    if left_index or right_index or right_on is not None or left_on is not None or on is not None:
        raise RuntimeError(
            "Can not pass on, right_on, left_on or set right_index=True or " "left_index=True"
        )

    cross_col = f"_cross_{uuid.uuid4()}"
    left = left.assign(**{cross_col: 1})
    right = right.assign(**{cross_col: 1})

    left_on = right_on = [cross_col]

    result = dx.merge(
        left,
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        suffixes=suffixes,
        npartitions=npartitions,
        indicator=indicator,
        # shuffle=shuffle,
        broadcast=broadcast,
    )
    del result[cross_col]
    return result


def dict_inter(dictionary, iterable):
    return {k: v for k, v in dictionary.items() if k in iterable}
