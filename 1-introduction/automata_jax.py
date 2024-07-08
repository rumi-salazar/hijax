"""
Elementary cellular automata simulator in jax.
"""


import functools
import itertools
import pathlib
import time
from typing import Literal

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
import einops


def main(
    rule: int = 110,
    width: int = 32,
    height: int = 32,
    init: Literal["random", "middle"] = "middle",
    seed: int = 42,
    animate: bool = True,
    fps: None | float = None,
    save_image: None | pathlib.Path = None,
    upscale: int = 1,
):
    # interpret rule
    rule_uint8 = jnp.uint8(rule)
    rule_8bits = jnp.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_8bits.reshape(2, 2, 2)
    print("rule:", rule_uint8)
    print("bits:", rule_8bits)
    print("table:")
    for a, b, c in itertools.product([0, 1], repeat=3):
        print(' ', a, b, c, "->", rule_table[a, b, c])

    # initialise the state
    match init:
        case "middle":
            state = jnp.zeros(width, dtype=jnp.uint8)
            state = state.at[width//2].set(1)
        case "random":
            key = jax.random.key(seed)
            state = jax.random.randint(
                key=key,
                minval=0,
                maxval=2, # not included
                shape=(width,),
                dtype=jnp.uint8,
            )
    print("initial state:")
    print(state, state.dtype)

    omnisim = jax.vmap(
        jax.jit(simulate, static_argnames=('height',)),
        in_axes=(0,None,None),
    )
    histories = omnisim(
        jnp.arange(256, dtype=jnp.uint8),
        state,
        height,
    )
    print(histories.shape)
    omnihistory = einops.rearrange(
        histories,
        '(r1 r2) h w -> (r1 h) (r2 w)',
        r1=16,
        r2=16,
    )
    # render to image
    if save_image is not None:
        print("rendering to", save_image, "...")
        omnihistory_greyscale = 255 * (1-omnihistory)
        omnihistory_upscaled = (omnihistory_greyscale
            .repeat(upscale, axis=0)
            .repeat(upscale, axis=1)
        )
        Image.fromarray(np.asarray(omnihistory_upscaled)).save(save_image)

    # # conduct the simulation
    # print("simulating...")
    # start_time = time.perf_counter()
    # history = jax.jit(simulate, static_argnames=('height',))(
    #     rule=rule_uint8,
    #     init_state=state,
    #     height=height,
    # )
    # end_time = time.perf_counter()
    # print("simulation complete!")
    # print("result shape", history.shape)
    # print("time taken", end_time - start_time, "seconds")

    # # render to screen
    # if animate:
    #     print("rendering...")
    #     for row in history:
    #         print(''.join(["█░"[s]*2 for s in row]))
    #         if fps is not None: time.sleep(1/fps)

    # # render to image
    # if save_image is not None:
    #     print("rendering to", save_image, "...")
    #     history_greyscale = 255 * (1-history)
    #     history_upscaled = (history_greyscale
    #         .repeat(upscale, axis=0)
    #         .repeat(upscale, axis=1)
    #     )
    #     Image.fromarray(history_upscaled).save(save_image)


def simulate(
    rule: jnp.uint8,
    init_state: jax.Array,  # uint8[width]
    height: int,
) -> jax.Array:             # uint8[height, width]
    # parse input
    rule_table = jnp.unpackbits(rule, bitorder='little').reshape(2,2,2)

    # extra width is to implement wraparound with slicing
    init_state = jnp.pad(init_state, 1, mode='wrap')

    # remaining rows
    # for step in tqdm.trange(1, height):
    def iterate(last_state, _input):
        # apply rules
        next_state = jnp.pad(
            rule_table[
                last_state[0:-2],
                last_state[1:-1],
                last_state[2:  ],
            ],
            1,
            mode='wrap',
        )
        return next_state, next_state
    _final_state, history_ = jax.lax.scan(
        iterate,
        init_state,
        jnp.arange(1, height),
    )
    history = jnp.concatenate((init_state.reshape(1,-1), history_), axis=0)

    return history[:, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

