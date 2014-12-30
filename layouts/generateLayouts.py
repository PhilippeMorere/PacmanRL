__author__ = 'philippe'
import random
import numpy

nb_layouts = 100
min_width = 6
max_width = 10
min_height = 6
max_height = 10
widths = range(min_width, max_width + 1)
heights = range(min_height, max_height + 1)

for layer_i in range(nb_layouts):
    width = widths[(int)(random.random() * len(widths))]
    height = heights[(int)(random.random() * len(heights))]
    nb_cell = (width - 2) * (height - 2)
    pacman_is_set = False
    map_layout = ""
    for line_i in range(height):
        line = ""
        if line_i == 0 or line_i == height - 1:
            line = "%" * width
        else:
            for col_i in range(width):
                if col_i == 0 or col_i == width - 1:
                    line += "%"
                else:
                    nb_cell -= 1
                    if nb_cell == 0 and not pacman_is_set:
                        line += "P"
                        continue
                    if not pacman_is_set:
                        if random.random() < 0.01:
                            line += "P"
                            pacman_is_set = True
                            continue
                    rand = random.random()
                    if rand < 0.03:
                        line += "G"
                    elif rand < 0.18:
                        line += "."
                    elif rand < 0.30:
                        line += "%"
                    else:
                        line += " "
        map_layout += line + "\n"

    # Add a ghost if there is none
    index = -1
    try:
        index = map_layout.index("G")
    except ValueError:
        pass
    if index == -1:
        m = list(map_layout)
        insert_at = map_layout.index(" ") + 1
        while insert_at < len(m) and m[insert_at] != " ":
            insert_at += 1
        if insert_at == len(m) - 1:
            insert_at = map_layout.index(" ")
        m[insert_at] = "G"
        map_layout = "".join(m)

    print map_layout
    f = open("generatedLayouts/layout" + str(layer_i) + ".lay", "w")
    f.write(str(map_layout))
    #numpy.savetxt(f, numpy.array(map_layout).T)