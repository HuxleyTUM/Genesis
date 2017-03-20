def correct_colour_value(val):
    return min(max(0, int(val)), 255)


def visualise_magnitude(magnitude):
    r_value = correct_colour_value(-magnitude*255)
    g_value = correct_colour_value(magnitude*255)
    return r_value, g_value, 0, 255


def grey(val):
    corrected = correct_colour_value(val)
    return corrected, corrected, corrected, 255


def r(val):
    corrected = correct_colour_value(val)
    return corrected, 0, 0, 255


def g(val):
    corrected = correct_colour_value(val)
    return 0, corrected, 0, 255


def b(val):
    corrected = correct_colour_value(val)
    return 0, 0, corrected, 255


def a(c, a):
    return c[0], c[1], c[2], a


def adjust_colour(colour, factor):
    return [correct_colour_value(x * factor) for x in colour[0:3]] + [colour[3]]

WHITE = grey(255)
BLACK = grey(0)
RED = r(255)
GREEN = g(255)
BLUE = b(255)
TEAL = (0, 255, 255, 255)
TRANSPARENT = a(BLACK, 0)
