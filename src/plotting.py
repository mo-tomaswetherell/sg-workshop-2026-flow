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
