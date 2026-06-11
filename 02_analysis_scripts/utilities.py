import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import os
from matplotlib import font_manager
import matplotlib
from cycler import cycler

plot_options_default = {
    "linewidth": 3,
    "fontsize": 22,
    "fontname": "serif",
    "markersize": 20,
    "axes_linewidth": 1,
    "showspines": True,
    "colorcycle": "1",
}

COLOR_CYCLES = {
    "1": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
    ]
}


def set_color_cycler(color_cycle):
    """Set matplotlib color cycle for plots.

    This function updates the matplotlib rcParams to use a custom color cycle
    for all subsequent plots. The color cycle determines the sequence of colors
    used when plotting multiple data series.

    Args:
        color_cycle (list): List of color strings (hex, named colors, etc.)
                           that define the color sequence for plotting.
                           Example: ["#FF0000", "#00FF00", "#0000FF"]

    Returns:
        None

    Example:
        >>> colors = ["#FF0000", "#00FF00", "#0000FF"]
        >>> set_color_cycler(colors)
        # All subsequent plots will cycle through red, green, blue
    """
    new_cycler = cycler(color=color_cycle)
    plt.rcParams["axes.prop_cycle"] = new_cycler


def default_color_cycles(cycle="1"):
    """Get and apply a predefined color cycle or custom color list.

    This function retrieves a predefined color cycle from the COLOR_CYCLES
    dictionary or accepts a custom list of colors. It automatically applies
    the color cycle to matplotlib and returns the color list.

    Args:
        cycle (str or list, optional): Either a string key for predefined
                                     color cycles in COLOR_CYCLES dictionary,
                                     or a list of color strings.
                                     Defaults to "1".

    Returns:
        list: List of color strings that were applied to matplotlib.

    Raises:
        TypeError: If cycle is neither a string nor a list.
        KeyError: If cycle string is not found in COLOR_CYCLES dictionary.

    Examples:
        >>> # Use predefined color cycle
        >>> colors = default_color_cycles("1")
        >>> print(colors[0])  # "#a6cee3"

        >>> # Use custom colors
        >>> custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        >>> colors = default_color_cycles(custom_colors)
        >>> print(colors)  # ["#FF0000", "#00FF00", "#0000FF"]
    """
    if isinstance(cycle, str):
        colors = COLOR_CYCLES[cycle]
    elif isinstance(cycle, list):
        colors = cycle
    else:
        raise TypeError("cycle must be either a string key or list of colors")
    set_color_cycler(colors)

    return colors


def plot_setup(opts=None) -> dict:
    """Setup default plotting environment with customizable options.

    This function configures matplotlib rcParams to establish a consistent
    plotting style. It sets font properties, line styles, axis formatting,
    tick parameters, legend styling, and color cycles. If no options are
    provided, it uses the global plot_options_default dictionary.

    Args:
        opts (dict, optional): Dictionary of plotting options to override defaults.
                              If None, uses plot_options_default.
                              Supported keys:
                              - fontname: Font family name (default: "sans-serif")
                              - fontsize: Font size for labels and text (default: 18)
                              - linewidth: Line width for plots (default: 4)
                              - markersize: Marker size for scatter plots (default: 18)
                              - axes_linewidth: Width of axis lines (default: 1)
                              - showspines: Show top/right spines (default: False)
                              - colorcycle: Color cycle identifier or list (default: "1")

    Returns:
        dict: The complete options dictionary used for plotting, including
              the "colors" key with the applied color cycle.

    Example:
        >>> # Use default settings
        >>> options = plot_setup()
        >>> print(options["fontsize"])  # 18

        >>> # Custom settings
        >>> custom_opts = {"fontsize": 14, "linewidth": 2}
        >>> options = plot_setup(custom_opts)
        >>> print(options["linewidth"])  # 2
        >>> print(options["colors"])  # List of colors from color cycle

    Note:
        This function modifies matplotlib's global rcParams, affecting all
        subsequent plots until changed again or matplotlib is reset.
    """

    if opts is None:
        opts = plot_options_default.copy()

    # font name
    plt.rcParams["font.family"] = opts.get("fontname", plot_options_default["fontname"])

    # line properties
    plt.rcParams["lines.linewidth"] = opts.get(
        "linewidth", plot_options_default["linewidth"]
    )
    plt.rcParams["lines.markersize"] = opts.get(
        "markersize", plot_options_default["markersize"]
    )

    # axis properties
    plt.rcParams["xtick.labelsize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )
    plt.rcParams["ytick.labelsize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )
    plt.rcParams["axes.labelsize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )
    plt.rcParams["axes.titlesize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )
    plt.rcParams["axes.linewidth"] = opts.get(
        "axes_linewidth", plot_options_default["axes_linewidth"]
    )
    plt.rcParams["xtick.major.width"] = opts.get(
        "axes_linewidth", plot_options_default["axes_linewidth"]
    )
    plt.rcParams["ytick.major.width"] = opts.get(
        "axes_linewidth", plot_options_default["axes_linewidth"]
    )
    plt.rcParams["xtick.minor.width"] = (
        opts.get("axes_linewidth", plot_options_default["axes_linewidth"]) * 0.8
    )
    plt.rcParams["ytick.minor.width"] = (
        opts.get("axes_linewidth", plot_options_default["axes_linewidth"]) * 0.8
    )
    plt.rcParams["axes.spines.right"] = opts.get(
        "showspines", plot_options_default["showspines"]
    )
    plt.rcParams["axes.spines.top"] = opts.get(
        "showspines", plot_options_default["showspines"]
    )

    # legend properties
    plt.rcParams["legend.fontsize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )

    plt.rcParams["legend.title_fontsize"] = opts.get(
        "fontsize", plot_options_default["fontsize"]
    )

    # Apply color cycle and add to options
    opts["colors"] = default_color_cycles(
        opts.get("colorcycle", plot_options_default["colorcycle"])
    )

    return opts


def save_figure(
    fig=None,
    plot_name="plot",
    save_flag=1,
    open_flag=1,
    file_format=["png"],
    folder_name=None,
    use_date=True,
    full_path=None,
) -> None:
    """Save matplotlib figure to file with flexible naming and directory options.

    This function provides a comprehensive way to save matplotlib figures with
    automatic directory creation, date prefixing, multiple format support,
    and optional file explorer opening. It handles path management and ensures
    directories exist before saving.

    Args:
        fig (matplotlib.figure.Figure, optional): Figure object to save.
                                                 If None, uses current figure (plt.gcf()).
                                                 Defaults to None.
        plot_name (str, optional): Base name for the saved file (without extension).
                                  Defaults to "plot".
        save_flag (int, optional): Whether to save the figure (1) or not (0).
                                  Defaults to 1.
        open_flag (int, optional): Whether to open the containing directory after saving (1) or not (0).
                                  Only works on Windows. Defaults to 1.
        file_format (list, optional): List of file formats to save.
                                     Supports any matplotlib-compatible format.
                                     Defaults to ["png"].
        folder_name (str, optional): Subdirectory name within the plots folder.
                                    If None, saves directly to plots folder.
                                    Defaults to None.
        use_date (bool, optional): Whether to prefix filename with current date (YYYY-MM-DD).
                                  Defaults to True.
        full_path (str or Path, optional): Complete path to save directory.
                                          If provided, overrides the default plots folder structure.
                                          Defaults to None.

    Returns:
        None

    Raises:
        OSError: If directory creation fails.
        IOError: If figure saving fails.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])

        >>> # Basic save with date prefix
        >>> save_figure(fig, "my_plot")
        # Saves as: ./plots/2024-01-15_my_plot.png

        >>> # Multiple formats, custom folder
        >>> save_figure(fig, "analysis", file_format=["png", "pdf"],
        ...             folder_name="results")
        # Saves as: ./plots/results/2024-01-15_analysis.png and .pdf

        >>> # Custom path, no date
        >>> save_figure(fig, "final_plot", full_path="./output/figures",
        ...             use_date=False, open_flag=0)
        # Saves as: ./output/figures/final_plot.png

    Note:
        - Creates directory structure automatically if it doesn't exist
        - Prints the full path of each saved file
        - Uses bbox_inches="tight" and facecolor="w" for clean output
        - Directory opening (open_flag) only works on Windows systems
    """

    if fig is None:
        fig = plt.gcf()

    # Include date in plot name if requested
    if use_date:
        now = datetime.datetime.now()
        dt_str = now.strftime("%Y-%m-%d")
        plot_name = f"{dt_str}_{plot_name}"

    # Determine save directory
    here_path = Path(os.getcwd())

    if full_path is not None:
        dir_path = Path(full_path)
    elif folder_name is not None:
        dir_path = here_path / "plots" / folder_name
    else:
        dir_path = here_path / "plots"

    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # Save figure in requested formats
    if save_flag:
        for format in file_format:
            save_path = dir_path / f"{plot_name}.{format}"
            fig.savefig(save_path, format=format, bbox_inches="tight", facecolor="w")
            print(f"Saved: {save_path}")

        # Open directory if requested (Windows only)
        if open_flag:
            try:
                os.startfile(dir_path)
            except (AttributeError, OSError):
                # AttributeError: not Windows, OSError: file explorer issues
                pass

    return


def format_plot(fig, p_opts=None) -> matplotlib.figure.Figure:
    """Apply consistent formatting to all axes in a matplotlib figure.

    This function iterates through all axes in a figure and applies consistent
    font sizing and styling to tick labels, axis labels, and titles. It provides
    a way to retroactively format plots that weren't created with the standard
    plot_setup() configuration.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object to format.
        p_opts (dict, optional): Formatting options dictionary. If None,
                               calls plot_setup() to get default options.
                               Expected to contain "font_size" key.

    Returns:
        matplotlib.figure.Figure: The same figure object with formatting applied.

    Raises:
        KeyError: If p_opts doesn't contain required "font_size" key.

    Examples:
        >>> fig, (ax1, ax2) = plt.subplots(1, 2)
        >>> ax1.plot([1, 2, 3], [1, 4, 2])
        >>> ax1.set_title("Plot 1")
        >>> ax2.scatter([1, 2, 3], [2, 3, 1])
        >>> ax2.set_title("Plot 2")

        >>> # Apply default formatting
        >>> formatted_fig = format_plot(fig)

        >>> # Apply custom formatting
        >>> custom_opts = {"font_size": 14}
        >>> formatted_fig = format_plot(fig, custom_opts)

    Note:
        - Handles exceptions gracefully - if formatting fails for any axis,
          it prints an error message but continues processing other axes
        - Modifies the figure in-place and also returns it
        - Requires p_opts to have "font_size" key, not "fontsize" like plot_setup()
    """

    # Load default options if not provided
    if p_opts is None:
        p_opts = plot_setup()

    # Get all axes from the figure
    axes = fig.axes

    # Apply formatting to each axis
    for i_ax, ax in enumerate(axes):
        try:
            # Update font sizes for all text elements
            ax.tick_params(axis="both", which="major", labelsize=p_opts["font_size"])
            ax.set_xlabel(ax.get_xlabel(), fontsize=p_opts["font_size"])
            ax.set_ylabel(ax.get_ylabel(), fontsize=p_opts["font_size"])
            ax.set_title(ax.get_title(), fontsize=p_opts["font_size"])

            # Assign back to figure (though this is technically unnecessary)
            fig.axes[i_ax] = ax

        except Exception as e:
            print(f"Could not format axis {i_ax}: {e}")

    return fig


def set_font_from_file(path_to_font) -> None:
    """Set matplotlib font from a font file (TTF, OTF, etc.).

    This function loads a font file and configures matplotlib to use it as
    the default sans-serif font. It's useful for using custom fonts or ensuring
    consistent typography across different systems.

    Args:
        path_to_font (str or Path): Path to the font file. Must be a valid
                                   font file (e.g., .ttf, .otf) that exists
                                   on the filesystem.

    Returns:
        None

    Raises:
        Exception: If the provided path doesn't exist.
        OSError: If the font file cannot be loaded by matplotlib.

    Examples:
        >>> # Use a custom TTF font
        >>> set_font_from_file("./fonts/MyCustomFont.ttf")
        Font Set Successfully.

        >>> # Use system font (Windows example)
        >>> set_font_from_file("C:/Windows/Fonts/arial.ttf")
        Font Set Successfully.

        >>> # This will raise an exception
        >>> set_font_from_file("./nonexistent_font.ttf")
        Exception: ./nonexistent_font.ttf is not a valid path.

    Note:
        - The font is set as the default sans-serif font in matplotlib
        - This affects all subsequent plots until matplotlib is restarted
          or the font is changed again
        - Font changes may not be visible in already-created figures,
          only in new plots
        - Prints success message when font is loaded successfully
    """

    if not os.path.exists(path_to_font):
        raise Exception(f"{str(path_to_font)} is not a valid path.")

    # Register the font with matplotlib's font manager
    font_manager.fontManager.addfont(path_to_font)

    # Create font properties object to get the font name
    prop = font_manager.FontProperties(fname=path_to_font)

    # Set matplotlib to use sans-serif family with this font as default
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()

    print("Font Set Successfully.")

    return



def set_min_ticks(ax, axes="y", which="major", num_ticks=5):
    """
    Set minimum number of ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify
    axes : which axes
        The axes to modify: x, y, both
    which : str, default='major'
        Which ticks to modify ('major' or 'minor')
    num_ticks : int, optional
        Minimum number of y-axis ticks

    """
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    if axes in ["x", "both"]:
        if which in ["major", "both"]:
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks)
            )
        if which in ["minor", "both"]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(num_ticks))

    if axes in ["y", "both"]:
        if which in ["major", "both"]:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=num_ticks, min_n_ticks=num_ticks)
            )
        if which in ["minor", "both"]:
            ax.yaxis.set_minor_locator(AutoMinorLocator(num_ticks))

    return ax
