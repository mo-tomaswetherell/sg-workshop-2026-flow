from pathlib import Path
import colormaps
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as mpl_anim
import math

plt.rcParams["font.family"] = "DejaVu Sans Mono"

# Custom discrete colormap registry.
# Each entry maps a name to bounds (mm/day) and a corresponding list of colors.
# len(colors) must equal len(bounds) - 1.
CUSTOM_CMAPS: dict[str, dict] = {
    "precip": {
        "bounds": [0, 0.1, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500],
        "colors": [
            "#FFFFFF",
            "#E5E5F9",
            "#D9D9FF",
            "#C4D4FF",
            "#8EB2FF",
            "#7F96FF",
            "#4AE2E8",
            "#B0F9C8",
            "#8FED8F",
            "#FFE48F",
            "#FFC686",
            "#FFA931",
            "#FF8025",
            "#F83D0E",
            "#E20202",
            "#B50000",
            "#940000",
        ],
    },
}

def resolve_cmap(
    cmap: str | object,
) -> tuple[object, mcolors.Normalize | None]:
    """Return (cmap_obj, norm) for any supported colormap specification.

    Accepts:
      - a colormap object (matplotlib or colormaps package) — returned as-is
      - a string name checked in order: CUSTOM_CMAPS → matplotlib → colormaps package
    """
    if not isinstance(cmap, str):
        return cmap, None

    if cmap in CUSTOM_CMAPS:
        spec = CUSTOM_CMAPS[cmap]
        cmap_obj = mcolors.ListedColormap(spec["colors"], name=cmap)
        norm = mcolors.BoundaryNorm(spec["bounds"], cmap_obj.N)
        return cmap_obj, norm

    try:
        return plt.get_cmap(cmap), None
    except (ValueError, KeyError):
        pass

    if hasattr(colormaps, cmap):
        return getattr(colormaps, cmap), None

    raise ValueError(f"Unknown colormap: '{cmap}'")

def plot_field(
    ds: xr.Dataset,
    time_idx: int,
    variable: str = "pr",
    cmap: str | object = "precip",
    vmin: float | None = None,
    vmax: float | None = None,
    units: str | None = None,
    colorbar: bool = True,
) -> None:
    """Plot a field.

    cmap can be:
      - a key from CUSTOM_CMAPS (e.g. "precip")
      - any matplotlib colormap name (e.g. "Blues")
      - any colormaps-package colormap name (e.g. "WhBlGrYeRe")
      - a colormap object
    vmin/vmax are ignored when a custom discrete cmap is used (bounds define the range).
    """
    cmap_obj, norm = resolve_cmap(cmap)

    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    data = ds[variable].isel(time=time_idx)

    if units == "mm day-1":
        data = data * 86400
        data.attrs["units"] = units
        data.attrs["long_name"] = "Precipitation"

    plot_kwargs = dict(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_obj, add_colorbar=colorbar)
    if colorbar:
        plot_kwargs["cbar_kwargs"] = {"pad": 0.15}
    if norm is not None:
        plot_kwargs["norm"] = norm
    else:
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

    data.plot(**plot_kwargs)
    ax.set_title("")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, alpha=0.05, color="black")

    plt.show()


def plot_predictors(
    ds: xr.Dataset,
    variables: list[str],
    time_idx: int = 0,
    cmap: str | object = "viridis",
    ncols: int = 3,
) -> None:
    """Plot a subset of predictor fields in a grid, one panel per variable.

    Each panel has its own colourbar, autoscaled to that variable, since the
    predictors represent different physical quantities with very different ranges.

    Args:
        ds: Dataset containing the predictor variables.
        variables: Names of the variables to plot, one panel each.
        time_idx: Index along the `time` dimension.
        cmap: Colourmap specification (see `resolve_cmap`), applied to every panel.
        ncols: Number of columns in the grid.
    """
    cmap_obj, norm = resolve_cmap(cmap)

    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
        layout="constrained",
    )

    for idx, ax in enumerate(axes.flat):
        if idx >= len(variables):
            ax.set_visible(False)
            continue

        variable = variables[idx]
        data = ds[variable].isel(time=time_idx)

        plot_kwargs = dict(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap_obj,
            add_colorbar=True,
            cbar_kwargs={"pad": 0.15, "shrink": 0.8},
        )
        if norm is not None:
            plot_kwargs["norm"] = norm

        data.plot(**plot_kwargs)
        ax.set_title(variable)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.05, color="black")

    plt.show()

def plot_target_vs_predictions(
    ds: xr.Dataset,
    time_idx: int,
    variable: str = "pr",
    input_idx: int = 0,
    n_samples: int = 1,
    cmap: str | object = "precip",
    vmin: float | None = None,
    vmax: float | None = None,
    max_sample_cols: int = 4,
) -> None:
    """Plot the target alongside generated samples for a single time step.

    The target is shown first, follwed by the first `n_samples` generated samples.
    All panels share one colourbar.

    Args:
        ds: Dataset containing `pred_{variable}` (with a `sample` dimension) and
            `target_{variable}`.
        time_idx: Index along the `time` dimension.
        variable: Varaible name without the `pred_`/`target_` prefix.
        input_idx: Index along the `input` dimension.
        n_samples: Number of generated samples to display. Must not exceed the number of
            samples in the dataset.
        cmap: Colourmap specifcation (see `resolve_cmap`).
        vmin/vmax: Colour limits. Ignored for custom discrete colourmaps. Defaults to the
            min/max across all plotted panels.
        max_sample_cols: Maximum number of generated sample panels per row before wrapping.
    """
    max_samples = ds.sizes["sample"]
    if not 1 <= n_samples <= max_samples:
        raise ValueError(f"n_samples must be between 1 and {max_samples}, got {n_samples}.")

    pred = ds[f"pred_{variable}"].isel(input=input_idx, time=time_idx, sample=slice(0, n_samples))
    target = ds[f"target_{variable}"].isel(input=input_idx, time=time_idx)

    units = ds[f"target_{variable}"].attrs.get("units", "")
    if units == "kg m-2 s-1":
        pred = pred * 86400
        target = target * 86400
        units = "mm day-1"

    cmap_obj, norm = resolve_cmap(cmap)

    plot_kwargs = dict(transform=ccrs.PlateCarree(), cmap=cmap_obj, add_colorbar=False)
    if norm is not None:
        plot_kwargs["norm"] = norm
    else:
        plot_kwargs["vmin"] = (vmin if vmin is not None else float(min(target.min(), pred.min())))
        plot_kwargs["vmax"] = (vmax if vmax is not None else float(max(target.max(), pred.max())))

    # Grid layout: column 0 holds only the target; samples occupy columns >=1.
    sample_cols = min(n_samples, max_sample_cols)
    ncols = sample_cols + 1
    nrows = math.ceil(n_samples / sample_cols)

    # (title, data, row, col) for each panel
    panels = [("Target", target, 0, 0)] + [
        (f"Sample {i}", pred.isel(sample=i), i // sample_cols, 1 + i % sample_cols)
        for i in range(n_samples)
    ]
    occupied = {(row, col) for _, _, row, col in panels}

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize = (4 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
        layout="constrained",
    )

    # Hide axes in column 0 below the target
    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in occupied:
                axes[row, col].set_visible(False)

    mappable = None
    for title, data, row, col in panels:
        ax = axes[row, col]
        mappable = data.plot(ax=ax, **plot_kwargs)
        ax.set_title(title)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        is_target = (row, col) == (0, 0)
        gl = ax.gridlines(draw_labels=is_target, alpha=0.05, color="black")
        if is_target:
            gl.top_labels = False
            gl.right_labels = False

    fig.colorbar(
        mappable,
        ax=axes,
        orientation="horizontal",
        location="bottom",
        shrink=0.5,
        aspect=40,
        pad=0.05,
        label=units,
    )
    fig.suptitle(str(target["time"].dt.strftime("%Y-%m-%d").values))

    plt.show()

def make_discrete_symmetric_norm(
    lim: float, n_levels: int, cmap: str
) -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Fixed, discrete, symmetric colour scale spanning [-lim, lim].

    Uses `n_levels` equal-width bins, plus an extra colour at each end for values beyond
    +-lim. An odd `n_levels` gives a centre bin containing zero, which is the middle colour
    of a diverging colourmap.
    """
    cmap_obj = plt.get_cmap(cmap, n_levels + 2)  # +2 for the two extension colours
    bounds = np.linspace(-lim, lim, n_levels + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N, extend="both")
    return cmap_obj, norm

def animate_sampling(
    states: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    physical_states: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    variable: str = "pr",
    target_idx: int = 0,
    units: str | None = None,
    alpha: float | None = None,
    date: str | None = None,
    cmap_state: str = "RdBu_r",
    cmap_velocity: str = "PuOr",
    cmap_physical: str | object = "precip",
    physical_ticks: list[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    state_lim: float | None = None,
    velocity_lim: float | None = None,
    n_levels: int = 15,
    out_path: str | None = None,
    fps: int = 10,
    frame_stride: int = 1,
    hold_frames: int = 15,
    dpi: int = 100,
) -> mpl_anim.FuncAnimation:
    """Animate how a single sample is generated by integrating the flow ODE.

    Four panels: the normalised state x_t, the predicted velocity v_t, the simulation
    progress (time t agaisnt step) and the physical state.

    Args:
        states: Normalised states, shape (n_steps + 1,num_targets, lat, lon)
        velocities: Predicted velocities, shape (n_steps, num_targets, lat, lon)
        times: Simulation times, shape (n_steps + 1,)
        physical_states: Physical (un-normalised) states, shape (n_steps + 1, num_targets, lat, lon)
        lat, lon: 1D coordinate arrays of the target grid
        variable: Variable name (used in titles)
        target_idx: Index of the target variable to visualise
        units: Units of `physical_states`. "kg m-2 s-1" is converted to mm day-1
        alpha: Time-etepping exponenet, used for legend label
        date: Date string for the title
        cmap_state: Colormap for the state panel
        cmap_velocity: Colormap for the velocity panel
        cmap_physical: Colormap for the physical state panel
        physical_ticks: Colourbar ticks for the physical panel. Defaults to all bounds for a
            discrete colourmap.
        vmin/vmax: Physical colour limits. Ignored for custome discrete colourmaps.
        state_lim: Fixed half-range of the normalised state colour scale. Defaults to the
            99.5th percentile of |x|, rounded up to the nearest 0.5.
        velocity_lim: As `state_lim`, for the velocity panel.
        n_levels: Number of discrete colour levels for the state and velocity panels (use an
            odd number so zero sits in a central bin).
        out_path: Path to save the animation (.gif)
        fps: Frames per second for the animation
        frame_stride: Stride between frames to include in the animation
        hold_frames: Number of frames to hold the final frame
        dpi: Resolution of the saved animation

    Returns:
        The animation. The figure is closed so it is not rendered twice in a notebook.
    """
    n_steps = len(times) - 1
    x = states[:, target_idx]  # shape (n_steps + 1, lat, lon)

    # repeat final velocity for the last frame
    v = np.concatenate([velocities[:, target_idx], velocities[-1:, target_idx]], axis=0)  # shape (n_steps + 1, lat, lon)

    phys = physical_states[:, target_idx]  # shape (n_steps + 1, lat, lon)

    if units == "kg m-2 s-1":
        phys = phys * 86400
        units = "mm day-1"

    def nice_limit(arr: np.ndarray, step: float = 0.5) -> float:
        return max(step, float(np.ceil(np.percentile(np.abs(arr), 99.5) / step) * step))

    x_lim = state_lim if state_lim is not None else nice_limit(x)
    v_lim = velocity_lim if velocity_lim is not None else nice_limit(v)

    x_cmap, x_norm = make_discrete_symmetric_norm(x_lim, n_levels, cmap_state)
    v_cmap, v_norm = make_discrete_symmetric_norm(v_lim, n_levels, cmap_velocity)

    phys_cmap, phys_norm = resolve_cmap(cmap_physical)

    phys_kwargs = dict(cmap=phys_cmap)
    if phys_norm is not None:
        phys_kwargs["norm"] = phys_norm
    else:
        phys_kwargs["vmin"] = (vmin if vmin is not None else float(phys[-1].min()))
        phys_kwargs["vmax"] = (vmax if vmax is not None else float(phys[-1].max()))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_x = fig.add_subplot(gs[0, 0], projection=proj)
    ax_v = fig.add_subplot(gs[0, 1], projection=proj)
    ax_t = fig.add_subplot(gs[1, 0])
    ax_phys = fig.add_subplot(gs[1, 1], projection=proj)

    def make_map(ax, data, title, label, extend="neither", ticks=None, **kwargs):
        mesh = ax.pcolormesh(lon, lat, data, transform=proj, shading="auto", **kwargs)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, alpha=0.05, color="black")
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(title)
        fig.colorbar(
            mesh, ax=ax, orientation="horizontal", pad=0.08, shrink=0.8,
            label=label, extend=extend, ticks=ticks, format="%g",
        )
        return mesh

    mesh_x = make_map(
        ax_x, x[0], r"Normalised state $x_t$", "normalised",
        extend="both", ticks=np.linspace(-x_lim, x_lim, 5), cmap=x_cmap, norm=x_norm,
    )
    mesh_v = make_map(
        ax_v, v[0], r"Predicted velocity $u_{\theta}(x_t,t;c)$", "normalised / unit time",
        extend="both", ticks=np.linspace(-v_lim, v_lim, 5), cmap=v_cmap, norm=v_norm,
    )
    mesh_phys = make_map(
        ax_phys, phys[0], f"Physical state ({variable})", units or "",
        ticks=physical_ticks, **phys_kwargs,
    )

    # Progress panel
    steps = np.arange(n_steps + 1)
    ax_t.plot(steps, steps / n_steps, color="grey", ls="--", lw=1, label=r"$\alpha=1$")
    label = rf"$\alpha={alpha:g}$" if alpha is not None else "time steps"
    ax_t.plot(steps, times, color="C0", lw=2, label=label)
    (trail,) = ax_t.plot([], [], color="C3", lw=3)
    (dot,) = ax_t.plot([], [], "o", color="C3", ms=10, zorder=5)
    ax_t.set_xlim(0, n_steps)
    ax_t.set_ylim(0, 1.02)
    ax_t.set_xlabel("Step")
    ax_t.set_ylabel("Simulation time $t$")
    ax_t.set_title("Simulation progress")
    ax_t.legend(loc="upper left")
    ax_t.grid(alpha=0.2)

    frames = list(range(0, n_steps + 1, max(frame_stride, 1)))
    if frames[-1] != n_steps:
        frames.append(n_steps)
    frames += [n_steps] * hold_frames

    def update(k: int):
        i = frames[k]
        mesh_x.set_array(x[i].ravel())
        mesh_v.set_array(v[i].ravel())
        mesh_phys.set_array(phys[i].ravel())
        dot.set_data([i], [times[i]])
        trail.set_data(steps[: i + 1], times[: i + 1])
        prefix = f"{date}   " if date else ""
        fig.suptitle(f"{prefix}step {i:>3d}/{n_steps}   t = {times[i]:.3f}")
        return mesh_x, mesh_v, mesh_phys, dot, trail

    anim = mpl_anim.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    if out_path is not None:
        out_path = str(Path(out_path).with_suffix(".gif"))
        writer = "pillow"
        anim.save(out_path, writer=writer, fps=fps, dpi=dpi)
        print(f"Animation saved to {out_path}")

    plt.close(fig)
    return anim
