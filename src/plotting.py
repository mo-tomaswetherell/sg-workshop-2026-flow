import colormaps
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        data *= 86400

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
        (f"(Sample {i}", pred.isel(sample=i), i // sample_cols, 1 + i % sample_cols)
        for i in range(n_samples)
    ]
    occupied = {(row, col) for _, _, row, col in panels

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

        gl = ax.gridline(draw_labels=True, alpha=0.05, colour="black")
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (row, col - 1) not in occupied # true for panels with nothing to the left
        gl.bottom_labels = (row + 1, col) not in occupied # true for panels with nothing below
        
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
    fig.suptitle(str(target["time"].dt.strftime(%Y-%m-%d").values))

    plt.show()
        
    

    
        
