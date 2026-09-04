def tile_bounds(width: int, height: int, tile_size: int = 512):
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            yield (x, y, min(x + tile_size, width), min(y + tile_size, height))
