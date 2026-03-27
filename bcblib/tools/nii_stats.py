import warnings


def simple_stats(image, method='mean', axis=3):
    """Deprecated. Use ``bcblib.imaging.stats.reduce_axis`` instead."""
    warnings.warn(
        "bcblib.tools.nii_stats.simple_stats is deprecated. "
        "Use bcblib.imaging.stats.reduce_axis instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.stats import reduce_axis
    return reduce_axis(image, method=method, axis=axis)
