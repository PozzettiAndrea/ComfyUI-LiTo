#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#

import os
import random
import time


import webdataset as wds




def shuffle_with_life(
    data,
    max_life: int = 4,
    bufsize: int = 1000,
    initial: int = 100,
    rng=None,
    seed: int = None,
):
    """
    Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    Additionally, we keep track of the remaining life
    of a sample from the data. We reinsert a sample
    into the buffer if it has remaining life.

    Args:
        data: Iterator to shuffle.
        max_life (int): number of times a sample can be used.
        bufsize (int): Buffer size for shuffling.
        initial (int): Initial buffer size before yielding.
        rng: Random number generator.
        seed: Seed for the random number generator.
        handler: Exception handler.

    Yields:
        Shuffled items from the input iterator.

    Ref: https://github.com/webdataset/webdataset/blob/e0953f9bba17b416d5792d5a263b171c266e78be/src/webdataset/filters.py#L332

    Usage:
        shuffle_with_life_func = partial(
            wds_utils.shuffle_with_life,
            # iter
            max_life=self.wds_sample_life,
            bufsize=self.wds_buffer_size,
            initial=0,
            rng=None,
            seed=None,
        )
        wdset = wset.compose(shuffle_with_life_func)
    """
    if seed is not None:
        assert rng is None
        rng = random.Random(seed)
    elif rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))

    assert max_life > 0, f"max_life must be > 0, got {max_life}"
    assert bufsize > 0, f"bufsize must be > 0, got {bufsize}"
    assert initial >= 0, f"initial must be >= 0, got {initial}"
    initial = min(initial, bufsize)

    buf = []
    for sample in data:
        buf.append([sample, max_life])

        # Fill up to `initial` before we start yielding
        if len(buf) < initial:
            continue

        # Emit samples until buffer is back to bufsize
        while len(buf) > bufsize:
            _sample, _life = wds.filters.pick(buf, rng)
            _life -= 1
            if _life > 0:
                buf.append([_sample, _life])
            yield _sample

        # currently initial <= buffer size <= bufsize
        # Always emit at least one sample per new sample ingested
        if len(buf) >= initial:
            _sample, _life = wds.filters.pick(buf, rng)
            _life -= 1
            if _life > 0:
                buf.append([_sample, _life])

            if len(buf) < bufsize:
                # draw an additional sample if buffer is smaller than bufsize
                try:
                    buf.append([next(data), max_life])  # skipcq: PYL-R1708
                except StopIteration:
                    pass
            yield _sample

    # Drain remaining buffer
    while len(buf) > 0:
        _sample, _life = wds.filters.pick(buf, rng)
        _life -= 1
        if _life > 0:
            buf.append([_sample, _life])
        yield _sample


