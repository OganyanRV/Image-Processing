from timeit import default_timer as timer


def measure_time(callable, *args, max_time=5, num_measures=10, percent_skip=20, **kwargs):
    t0 = timer()
    result = callable(*args, **kwargs)
    time = timer() - t0

    if time > max_time:
        return (time, result)

    times = []
    for i in range(num_measures):
        t0 = timer()
        _ = callable(*args, **kwargs)
        times.append(timer() - t0)

    le = int(num_measures*percent_skip/100)
    ri = num_measures - le
    cnt = ri - le + 1
    sum = 0
    while le <= ri:
        sum += times[le]
        le += 1

    return (sum / cnt, result)
