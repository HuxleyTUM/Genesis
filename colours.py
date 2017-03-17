def correct_colour_value(val):
    return min(max(0, int(val)), 255)


def visualise_magnitude(magnitude):
    r_value = correct_colour_value(-magnitude*255)
    g_value = correct_colour_value(magnitude*255)
    return r_value, g_value, 0, 0


def rgb(val):
    corrected = correct_colour_value(val)
    return corrected, corrected, corrected, 0


def r(val):
    corrected = correct_colour_value(val)
    return corrected, 0, 0, 0


def g(val):
    corrected = correct_colour_value(val)
    return 0, corrected, 0, 0


def b(val):
    corrected = correct_colour_value(val)
    return 0, 0, corrected, 0


def adjust_colour(colour, factor):
    return [correct_colour_value(x * factor) for x in colour[:3]] + [0]

WHITE = rgb(255)
BLACK = rgb(0)
RED = r(255)
GREEN = b(255)
BLUE = g(255)
