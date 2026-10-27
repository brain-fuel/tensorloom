"""Construct bounded shape requests independently of inference; shrinking stays bounded."""

from hypothesis import strategies as st
from . import law_domain as D


def shapes():
    return st.builds(
        D.ShapeCase,
        operation=st.sampled_from(
            [
                getattr(D, n)()
                for n in (
                    "BroadcastShape",
                    "MatmulShape",
                    "ReshapeShape",
                    "SumShape",
                    "MeanShape",
                    "MaximumShape",
                    "MinimumShape",
                    "ExpShape",
                    "SqrtShape",
                    "DivideShape",
                )
            ]
        ),
        left=st.lists(st.integers(0, 4), max_size=3),
        right=st.lists(st.integers(0, 4), max_size=3),
        target=st.lists(st.integers(-1, 4), max_size=3),
        dtype=st.sampled_from(
            [D.ShapeFloat32(), D.ShapeFloat64(), D.ShapeInt64(), D.ShapeBool()]
        ),
        axis=st.sampled_from([D.AllAxes(), D.FirstAxis(), D.LastAxis(), D.EmptyAxes()]),
    )
