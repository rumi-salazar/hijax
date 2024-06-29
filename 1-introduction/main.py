"""
Elementary cellular automata simulator in numpy.
"""

import itertools
import time
from typing import Literal, Optional

import numpy as np
from PIL import Image


def main(
    rule: np.uint8 = 110,
    width: int = 32,
    height: int = 32,
    init: Literal["random", "middle"] = "middle",
    seed: int = 42,
    animate: bool = True,
    fps: None | float = None,
    save_image: None | str = None,
    upscale: int = 1,
):
    # interpret rule
    rule_uint8 = np.uint8(rule)
    rule_8bits = np.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_8bits.reshape(2,2,2)
    
    print("rule", rule_uint8)
    print("bits", rule_8bits)
    print("table:")
    for a, b, c in itertools.product([0, 1], repeat=3):
        print(' ', a, b, c, "->", rule_table[a, b, c])

    # initialise the state
    match init:
        case "middle":
            state = np.zeros(width, dtype=np.uint8)
            state[width//2] = 1
        case "random":
            np.random.seed(seed)
            state = (np.random.rand(width) < 0.5).astype(np.uint8)
    print("initial state")
    print(state)

    # conduct the simulation
    print("simulating...")
    history = simulate(
        rule_table=rule_table,
        init_state=state,
        height=height,
    )
    print("simulation complete! result shape", history.shape)

    # todo: profile? tqdm?

    # render to screen
    if animate:
        print("rendering...")
        for row in history:
            print(''.join(["█░"[s]*2 for s in row]))
            if fps is not None: time.sleep(1/fps)

    # render to image
    if save_image is not None:
        print("rendering to", save_image, "...")
        history_greyscale = 255 * (1-history)
        history_upscaled = (history_greyscale
            .repeat(upscale, axis=0)
            .repeat(upscale, axis=1)
        )
        Image.fromarray(history_upscaled).save(save_image)

        
def simulate(
    rule_table: np.typing.NDArray,
    init_state: np.typing.NDArray,
    height: int,
) -> np.typing.NDArray:
    # TODO: parse rule here?

    (width,) = init_state.shape
    history = np.zeros((height, width+2), dtype=np.uint8)

    # first row
    history[0, 1:-1] = init_state
    history[0, 0] = init_state[-1]
    history[0, -1] = init_state[0]
    
    # remaining rows
    for step in range(1, height):
        # apply rules
        history[step, 1:-1] = rule_table[
            history[step-1, 0:-2],
            history[step-1, 1:-1],
            history[step-1, 2:],
        ]
        # sync edges
        history[step, 0] = history[step, -2]
        history[step, -1] = history[step, 1]

    return history[:, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

