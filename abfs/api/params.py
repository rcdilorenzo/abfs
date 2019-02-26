from toolz import juxt, partial, curry, compose
from lenses import lens

constrain = lambda low, high: compose(partial(min, high), partial(max, low))

class Params():
    coordinate = lens['coordinate']

    coordinates = juxt([
        coordinate['latitude'].get(),
        coordinate['longitude'].get(),
        compose(constrain(0, 1), lens.Get('tolerance', 0.5).get()),
        compose(constrain(10, 19), int, lens.Get('zoom', 16).get())
    ])

    image_url = lens['image_url'].get()
