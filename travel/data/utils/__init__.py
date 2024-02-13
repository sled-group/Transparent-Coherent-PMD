def generate_float_series(start: float, end: float, step: float) -> list[float]:
    """
    Generates a list of floats including a `start` and `end` point, and intermediate points `step` apart.

    :param start: Starting point.
    :param end: Ending point.
    :param step: Interval to generate series of floats between `start` and `end`.
    :return: List of floats sampling from the interval between `start` and `end`, each `step` apart.
    """
    # Ensure step is a positive float
    step = abs(step)

    # Initialize the series with the start value
    series = [start]

    # Generate numbers in the series
    while start + step <= end:
        start += step
        series.append(start)

    # Check if the end value is already in the series
    if series[-1] != end:
        series.append(end)

    return series
