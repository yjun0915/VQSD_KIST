import time
from contextlib import contextmanager
from TimeTagger import createTimeTagger, Resolution_Standard, Coincidences, Counter, freeTimeTagger


@contextmanager
def timetagger_session(cw, binwidth, n_value, delay):
    start = time.time()
    tagger = createTimeTagger(resolution=Resolution_Standard)
    coincidences = Coincidences(
        tagger=tagger,
        coincidenceGroups=[[1, 2]],
        coincidenceWindow=cw,
    )
    counter = Counter(
        tagger=tagger,
        channels=[1, 2, list(coincidences.getChannels())[0]],
        binwidth=binwidth * 1e9,
        n_values=n_value,
    )
    tagger.setInputDelay(channel=1, delay=delay[0])
    tagger.setInputDelay(channel=2, delay=delay[1])
    yield counter
    end = time.time()
    print("[TimeTagger]", end - start)
    freeTimeTagger(tagger=tagger)
