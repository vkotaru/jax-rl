"""Drawing a cart-pole trajectory. Specific to this env, not general.

Angle convention follows the env: theta = 0 is hanging, theta = pi is upright,
positive counter-clockwise. The bob sits at (x + l sin(th), -l cos(th)).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection


def bob(state, l):
    """Pivot and bob positions for one state."""
    x, th = float(state[0]), float(state[1])
    return (x, 0.0), (x + l * np.sin(th), -l * np.cos(th))


def _setup(ax, states, l):
    xs = np.asarray(states)[:, 0]
    pad = l * 1.4
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.axhline(0.0, lw=1, color=GRID, zorder=0)
    ax.axvline(0.0, lw=1, ls=":", color=GRID, zorder=0)   # x = 0 reference
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# palette: teal rod, coral bob, deep-green cart on warm white
BG   = "#fbfbf9"
GRID = "#e2e9e6"
ROD  = "#2f6f6a"
BALL = "#ff7043"
CART = "#1f3a37"
TEXT = "#7f938f"


def frame(ax, state, l, alpha=1.0, color=ROD, ball_color=BALL):
    """Draw one pose onto an existing axis."""
    (px, py), (bx, by) = bob(state, l)
    ax.plot([px, bx], [py, by], lw=3, color=color, alpha=alpha, zorder=2)
    ax.add_patch(plt.Rectangle((px - 0.12, -0.05), 0.24, 0.10,
                               color=CART, alpha=alpha, zorder=3))
    ax.plot([bx], [by], "o", ms=9, color=ball_color, alpha=alpha, zorder=4)


def strip(states, l, path, n=8, title=None):
    """A row of still poses -- quick to eyeball, no animation needed."""
    states = np.asarray(states)
    idx = np.linspace(0, len(states) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        _setup(ax, states, l)
        frame(ax, states[i], l)
        ax.set_title(f"t={i}", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate(states, l, path, fps=50, stride=1, title=None, dt=None,
             trail_len=40):
    """Animated GIF of a trajectory. Pass dt to stamp the simulated time."""
    states = np.asarray(states)
    times = np.arange(len(states)) * dt if dt is not None else None
    states = states[::stride]
    if times is not None:
        times = times[::stride]

    fig, ax = plt.subplots(figsize=(5, 4))
    _setup(ax, states, l)
    if title:
        ax.set_title(title, fontsize=10)
    rod, = ax.plot([], [], lw=3, color=ROD, zorder=2)
    tip, = ax.plot([], [], "o", ms=9, color=BALL, zorder=4)
    cart = plt.Rectangle((0, 0), 0.24, 0.10, color=CART, zorder=3)
    ax.add_patch(cart)

    # fading trail of where the ball has been
    track = np.array([bob(st, l)[1] for st in states])
    trail = LineCollection([], linewidths=2.0, zorder=1)
    ax.add_collection(trail)
    rgb = matplotlib.colors.to_rgb(BALL)
    clock = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top",
                    fontsize=10, family="monospace", color=TEXT)

    def update(i):
        (px, py), (bx, by) = bob(states[i], l)
        rod.set_data([px, bx], [py, by])
        tip.set_data([bx], [by])
        cart.set_xy((px - 0.12, -0.05))
        if times is not None:
            clock.set_text(f"t = {times[i]:6.2f} s")

        j = max(0, i - trail_len)
        pts = track[j:i + 1]
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            a = np.linspace(0.0, 0.65, len(segs))       # oldest faint, newest solid
            trail.set_segments(segs)
            trail.set_color([(*rgb, ai) for ai in a])
        else:
            trail.set_segments([])
        return rod, tip, cart, clock, trail

    anim = FuncAnimation(fig, update, frames=len(states), blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _screen_inches(default=(16.0, 9.0)):
    """Screen size in inches, or a sensible default if it cannot be queried."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        px = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        dpi = plt.rcParams["figure.dpi"]
        return (px[0] / dpi, px[1] / dpi)
    except Exception:
        return default


def compare_grid(obs_by_name, l, n=None, dt=None, stride=4, fps=50,
                 alphas=None, colors=None, show=True, pole_scale=1.0,
                 screen_frac=0.5, figsize=None, trail_len=60, save=None):
    """Animate several episodes side by side, all policies on the same axes.

    `obs_by_name` maps a label to a batched obs array of shape
    (n_episodes, horizon, 4). Episode i of every series is drawn in panel i,
    so if the series were rolled out from the same key they start identically
    and any divergence is the policy, not the initial condition.

    `pole_scale` lengthens the drawn rod without touching the physics -- the
    cart still moves over metres while the pole is half a metre, so at true
    scale it is a sliver. Cosmetic only; do not use it for a figure where the
    geometry is the point.

    The window defaults to `screen_frac` of each screen dimension, with the
    content aspect preserved inside that box. Pass `figsize` to override.

    `trail_len` frames of the bob's recent path are drawn behind it, fading
    out, so the shape of a divergence is visible in a still frame.

    Pass `save` to write a GIF instead of (or as well as) showing the window.

    Returns the FuncAnimation. Keep the reference -- matplotlib stops the
    animation as soon as it is garbage collected.
    """
    names = list(obs_by_name)
    arrs = [np.asarray(obs_by_name[k])[:, ::stride] for k in names]
    n_eps = min(a.shape[0] for a in arrs) if n is None else min(
        n, min(a.shape[0] for a in arrs))
    arrs = [a[:n_eps] for a in arrs]
    frames = min(a.shape[1] for a in arrs)

    if alphas is None:                      # earlier series ghosted
        alphas = [0.28] * (len(names) - 1) + [1.0]
    if colors is None:
        colors = [ROD] * len(names)

    cols = int(np.ceil(np.sqrt(n_eps)))
    rows = int(np.ceil(n_eps / cols))

    # One x-window for every panel, so panels are comparable and the figure can
    # be sized to match: with aspect="equal" a mismatched box is all whitespace.
    span = max(float(np.abs(np.concatenate([a[:, :, 0].ravel() for a in arrs])).max()),
               0.6) + l * 1.3
    draw_l = l * pole_scale
    yspan = draw_l * 2.7

    if figsize is None:
        # fit the content aspect inside screen_frac of the screen
        content = (2 * span * cols, yspan * rows)
        box_w, box_h = (d * screen_frac for d in _screen_inches())
        scale = min(box_w / content[0], (box_h - 0.8) / content[1])
        figsize = (content[0] * scale, content[1] * scale + 0.8)

    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             facecolor=BG, squeeze=False)
    axes = axes.ravel()

    artists, clocks = [], []
    for e in range(n_eps):
        ax = axes[e]
        ax.set_facecolor(BG)
        ax.set_xlim(-span, span); ax.set_ylim(-draw_l * 1.35, draw_l * 1.35)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, lw=1, color=GRID, zorder=0)
        ax.axvline(0, lw=1, ls=":", color=GRID, zorder=0)

        per_series = []
        for si, name in enumerate(names):
            a = alphas[si]
            trail = LineCollection([], linewidths=1.8, zorder=1 + si)
            ax.add_collection(trail)
            rod, = ax.plot([], [], lw=3, color=colors[si], alpha=a,
                           solid_capstyle="round", zorder=2 + si)
            tip, = ax.plot([], [], "o", ms=8, color=BALL, alpha=a, zorder=4 + si)
            cart = plt.Rectangle((0, 0), 0.22, 0.09, color=CART, alpha=a, zorder=3 + si)
            ax.add_patch(cart)
            per_series.append((rod, tip, cart, trail))
        artists.append(per_series)
        clocks.append(ax.text(.02, .06, "", transform=ax.transAxes, va="bottom",
                              fontsize=8, family="monospace", color=TEXT))

    for e in range(n_eps, len(axes)):
        axes[e].axis("off")

    handles = [plt.Line2D([], [], color=colors[i], lw=3, alpha=alphas[i], label=n_)
               for i, n_ in enumerate(names)]
    fig.legend(handles=handles, loc="upper center", ncol=len(names),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    # where the bob has been, per episode and series, for the fading trail
    tracks = [[np.array([bob(a[e, f], draw_l)[1] for f in range(frames)])
               for a in arrs] for e in range(n_eps)]
    rgb = matplotlib.colors.to_rgb(BALL)

    def update(f):
        out = []
        for e in range(n_eps):
            for si, a in enumerate(arrs):
                (px, py), (bx, by) = bob(a[e, f], draw_l)
                rod, tip, cart, trail = artists[e][si]
                rod.set_data([px, bx], [py, by])
                tip.set_data([bx], [by])
                cart.set_xy((px - 0.11, -0.045))

                pts = tracks[e][si][max(0, f - trail_len):f + 1]
                if len(pts) > 1:
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    fade = np.linspace(0.0, 0.7 * alphas[si], len(segs))
                    trail.set_segments(segs)
                    trail.set_color([(*rgb, al) for al in fade])
                else:
                    trail.set_segments([])
                out += [rod, tip, cart, trail]
            if dt is not None:
                clocks[e].set_text(f"t={f * stride * dt:5.2f}s")
                out.append(clocks[e])
        return out

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    if save is not None:
        anim.save(save, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    return anim
