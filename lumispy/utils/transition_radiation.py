# -*- coding: utf-8 -*-
# Copyright 2019-2026 The LumiSpy developers
#
# This file is part of LumiSpy.
#
# LumiSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the license, or
# (at your option) any later version.
#
# LumiSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LumiSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

from importlib.resources import as_file, files
import warnings

import numpy as np
from hyperspy import axes as hs_axes
from hyperspy.signals import Signal1D, Signal2D
from scipy.constants import c, e, h

__all__ = ["transition_radiation_nm"]


# Vacuum h*c in eV nm.
_HC_EV_NM = h * c / e * 1e9

# Atomic-unit constants.
_C_AU = 137.03599971
_AU_ENERGY_EV = 27.2113834

_DEFAULT_AXIS_SIZE = 200

_OPTICAL_CONSTANTS = (
    files("lumispy.data").joinpath("transition_radiation").joinpath("optical_constants")
)

_DATASETS = {
    "Al_mcpeak": {
        "layout": "separate_n_k",
        "representation": "n_k",
        "n_file": "mcpeak_n.csv",
        "k_file": "mcpeak_k.csv",
        "loadfile_kwargs": {"delimiter": ",", "skiprows": 1},
        "coordinate": "wavelength_um",
        "coordinate_column": 0,
        "n_column": 1,
        "k_column": 1,
    },
    "Al_rakic": {
        "layout": "separate_n_k",
        "representation": "n_k",
        "n_file": "rakic_n.csv",
        "k_file": "rakic_k.csv",
        "loadfile_kwargs": {"delimiter": ",", "skiprows": 1},
        "coordinate": "wavelength_um",
        "coordinate_column": 0,
        "n_column": 1,
        "k_column": 1,
    },
    "Ag_johnson": {
        "layout": "combined_n_k",
        "representation": "n_k",
        "file": "Ag_johnson.csv",
        "loadfile_kwargs": {"delimiter": ",", "skiprows": 1},
        "coordinate": "wavelength_um",
        "coordinate_column": 0,
        "n_column": 1,
        "k_coordinate_column": 2,
        "k_column": 3,
    },
    "Au_johnson": {
        "layout": "combined_n_k",
        "representation": "n_k",
        "file": "Au_johnson.txt",
        "loadfile_kwargs": {},
        "coordinate": "wavelength_um",
        "coordinate_column": 0,
        "n_column": 1,
        "k_column": 2,
    },
    "Au_palik": {
        "layout": "combined_epsilon",
        "representation": "epsilon",
        "file": "Au_palik1.txt",
        "loadfile_kwargs": {},
        "coordinate": "energy_ev",
        "coordinate_column": 0,
        "real_column": 1,
        "imaginary_column": 2,
    },
    "Au_werner": {
        "layout": "combined_n_k",
        "representation": "n_k",
        "file": "Au_werner.csv",
        "loadfile_kwargs": {"skiprows": 1},
        "coordinate": "wavelength_um",
        "coordinate_column": 0,
        "n_column": 1,
        "k_column": 2,
    },
    "Au_sc": {
        "layout": "combined_n_k",
        "representation": "n_k",
        "file": "Au_sc.dat",
        "loadfile_kwargs": {},
        "coordinate": "energy_ev",
        "coordinate_column": 0,
        "n_column": 1,
        "k_column": 2,
    },
}


def transition_radiation_nm(
    electron_energy=30.0,
    material_dataset="Al_mcpeak",
    angular_axis=(0.0, 90.0),
    spectral_axis=(150.0, 1200.0),
    return_angle_resolved=False,
):
    """Models probability per incoming electron of transition radiation in a given metal per unit nanometer.

    Parameters
    ----------
    electron_energy : float, optional
        Kinetic energy of the incident electron in keV. The default is 30 keV.
    material_dataset : str, optional
        Name of a bundled optical-constants dataset. Supported values are
        ``"Al_mcpeak"``, ``"Al_rakic"``, ``"Ag_johnson"``,
        ``"Au_johnson"``, ``"Au_palik"``, ``"Au_werner"``, and
        ``"Au_sc"``.
    angular_axis : tuple, array-like, or hyperspy.axes.BaseDataAxis, optional
        Emission-angle axis in degrees. A two-element tuple is interpreted as
        ``(start, stop)`` and creates 200 points. A three-element tuple is
        interpreted as ``(start, stop, number_of_points)``. Other array-like
        inputs are used as explicit axis values.
    spectral_axis : tuple, array-like, or hyperspy.axes.BaseDataAxis, optional
        Wavelength axis in nm. A two-element tuple is interpreted as
        ``(start, stop)`` and creates 200 points. A three-element tuple is
        interpreted as ``(start, stop, number_of_points)``. Other array-like
        inputs are used as explicit axis values.
    return_angle_resolved : bool, optional
        If ``False``, return only the angle-integrated spectrum. If ``True``,
        return ``(spectrum, angle_resolved)``.

    Returns
    -------
    hyperspy.signals.Signal1D
        Angle-integrated emission-probability density
    tuple of hyperspy.signals.Signal1D and hyperspy.signals.Signal2D
        Returned when ``return_angle_resolved=True``. The second signal is the
        emission-probability density per unit solid angle
    """
    angles_deg = _axis_to_numpy(angular_axis)
    wavelengths_nm = _axis_to_numpy(spectral_axis)

    (
        optical_coordinates,
        coordinate_kind,
        representation,
        first_optical_component,
        second_optical_component,
    ) = _load_optical_data(material_dataset)

    _validate_spectral_coverage(
        wavelengths_nm,
        optical_coordinates,
        coordinate_kind=coordinate_kind,
        material_dataset=material_dataset,
    )

    permittivity = _interpolate_permittivity(
        wavelengths_nm,
        optical_coordinates,
        first_optical_component,
        second_optical_component,
        coordinate_kind=coordinate_kind,
        representation=representation,
        material_dataset=material_dataset,
    )

    probability_nm, angular_probability_nm_sr = _calculate_transition_radiation(
        angles_deg,
        wavelengths_nm,
        electron_energy,
        permittivity,
    )

    common_metadata = {
        "Transition_radiation": {
            "electron_energy_keV": electron_energy,
            "material_dataset": material_dataset,
            "minimum_angle_deg": float(angles_deg[0]),
            "maximum_angle_deg": float(angles_deg[-1]),
        }
    }

    spectrum = Signal1D(
        probability_nm,
        axes=[
            {
                "axis": wavelengths_nm,
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
            }
        ],
        metadata={
            "General": {
                "title": (
                    f"{material_dataset} transition-radiation spectrum at "
                    f"{electron_energy:g} keV"
                )
            },
            **common_metadata,
        },
    )

    if not return_angle_resolved:
        return spectrum

    angle_resolved = Signal2D(
        angular_probability_nm_sr,
        axes=[
            {
                "axis": angles_deg,
                "name": "Angle",
                "units": "deg",
                "navigate": False,
            },
            {
                "axis": wavelengths_nm,
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
            },
        ],
        metadata={
            "General": {
                "title": (
                    f"{material_dataset} transition-radiation angular "
                    f"distribution at {electron_energy:g} keV"
                )
            },
            **common_metadata,
        },
    )

    return spectrum, angle_resolved


def _axis_to_numpy(axis):
    """Convert an axis to a numpy ndarray."""
    if isinstance(axis, hs_axes.BaseDataAxis):
        axis = np.asarray(axis.axis, dtype=float)

    elif isinstance(axis, tuple):
        if len(axis) == 2:
            start, stop = axis
            number_of_points = _DEFAULT_AXIS_SIZE

        elif len(axis) == 3:
            start, stop, number_of_points = axis

            try:
                number_of_points = int(number_of_points)
            except TypeError as error:
                raise TypeError(f"`{number_of_points}` must be an integer.") from error

            if number_of_points < 2:
                raise ValueError(f"`{axis}` needs at least two points.")

        else:
            raise ValueError(
                f"`{axis}` must be (start, stop), "
                "(start, stop, number_of_points), or an explicit axis."
            )

        axis = np.linspace(start, stop, number_of_points, dtype=float)

    else:
        try:
            axis = np.asarray(axis, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"`{axis}` must be an array-like object.") from error

    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"`{axis}` must contain at least two one-dimensional values.")

    if not np.all(np.isfinite(axis)):
        raise ValueError(f"`{axis}` must contain only finite values.")

    if not np.all(np.diff(axis) > 0):
        raise ValueError(f"`{axis}` must be strictly increasing.")

    return axis


def _load_optical_data(material_dataset):
    """Load optical data.

    Returns the coordinate values, coordinate kind, representation, and two
    real-valued optical components. For ``representation == 'n_k'`` the two
    components are ``n`` and ``k``. For ``representation == 'epsilon'`` they
    are the real and imaginary parts of the relative permittivity.
    """
    try:
        dataset = _DATASETS[material_dataset]
    except (KeyError, TypeError) as error:
        supported = ", ".join(sorted(_DATASETS))
        raise ValueError(
            f"Unknown material dataset {material_dataset!r}. "
            f"Supported datasets: {supported}."
        ) from error

    if dataset["layout"] == "separate_n_k":
        n_data = _loadfile(
            dataset["n_file"],
            **dataset["loadfile_kwargs"],
        )
        k_data = _loadfile(
            dataset["k_file"],
            **dataset["loadfile_kwargs"],
        )

        required_column = max(
            dataset["coordinate_column"],
            dataset["n_column"],
            dataset["k_column"],
        )
        _validate_data_columns(n_data, required_column, dataset["n_file"])
        _validate_data_columns(k_data, required_column, dataset["k_file"])

        if n_data.shape[0] != k_data.shape[0]:
            raise ValueError(
                f"The n and k datasets for {material_dataset!r} have different lengths."
            )

        n_coordinates = n_data[:, dataset["coordinate_column"]]
        k_coordinates = k_data[:, dataset["coordinate_column"]]

        if not np.allclose(n_coordinates, k_coordinates, rtol=1e-12, atol=0):
            raise ValueError(
                f"The n and k coordinate grids for {material_dataset!r} differ."
            )

        coordinates = n_coordinates
        first_component = n_data[:, dataset["n_column"]]
        second_component = k_data[:, dataset["k_column"]]

    elif dataset["layout"] == "combined_n_k":
        data = _loadfile(
            dataset["file"],
            **dataset["loadfile_kwargs"],
        )

        required_columns = [
            dataset["coordinate_column"],
            dataset["n_column"],
            dataset["k_column"],
        ]

        if "k_coordinate_column" in dataset:
            required_columns.append(dataset["k_coordinate_column"])

        _validate_data_columns(data, max(required_columns), dataset["file"])

        coordinates = data[:, dataset["coordinate_column"]]
        first_component = data[:, dataset["n_column"]]
        second_component = data[:, dataset["k_column"]]

        if "k_coordinate_column" in dataset:
            k_coordinates = data[:, dataset["k_coordinate_column"]]

            if not np.allclose(coordinates, k_coordinates, rtol=1e-12, atol=0):
                raise ValueError(
                    f"The n and k coordinate grids for {material_dataset!r} differ."
                )

    elif dataset["layout"] == "combined_epsilon":
        data = _loadfile(
            dataset["file"],
            **dataset["loadfile_kwargs"],
        )

        required_columns = [
            dataset["coordinate_column"],
            dataset["real_column"],
            dataset["imaginary_column"],
        ]
        _validate_data_columns(data, max(required_columns), dataset["file"])

        coordinates = data[:, dataset["coordinate_column"]]
        first_component = data[:, dataset["real_column"]]
        second_component = data[:, dataset["imaginary_column"]]

    else:
        raise ValueError(
            f"Unknown dataset layout {dataset['layout']!r} for {material_dataset!r}."
        )

    coordinates = np.asarray(coordinates, dtype=float)
    first_component = np.asarray(first_component, dtype=float)
    second_component = np.asarray(second_component, dtype=float)

    if not np.all(np.isfinite(coordinates)) or np.any(coordinates <= 0):
        raise ValueError(
            f"The coordinates for {material_dataset!r} must be positive and finite."
        )

    if not np.all(np.isfinite(first_component)):
        raise ValueError(
            f"The first optical-data component for {material_dataset!r} "
            "contains non-finite values."
        )

    if not np.all(np.isfinite(second_component)):
        raise ValueError(
            f"The second optical-data component for {material_dataset!r} "
            "contains non-finite values."
        )

    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    first_component = first_component[order]
    second_component = second_component[order]

    if not np.all(np.diff(coordinates) > 0):
        raise ValueError(
            f"The optical-data coordinates for {material_dataset!r} must be unique."
        )

    return (
        coordinates,
        dataset["coordinate"],
        dataset["representation"],
        first_component,
        second_component,
    )


def _validate_data_columns(data, maximum_column, filename):
    """Validate that a loaded table contains all configured columns."""
    if data.ndim != 2 or data.shape[1] <= maximum_column:
        raise ValueError(
            f"The optical-constants file {filename!r} does not contain all "
            "configured columns."
        )


def _loadfile(filename, **kwargs):
    """Load a bundled data file independently of the working directory."""
    resource = _OPTICAL_CONSTANTS.joinpath(filename)

    if not resource.is_file():
        raise FileNotFoundError(
            f"The bundled optical-constants file {filename!r} was not found."
        )

    with as_file(resource) as path:
        try:
            data = np.loadtxt(path, **kwargs)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not load optical-constants file {filename!r}."
            ) from error

    return np.atleast_2d(data)


def _wavelength_nm_to_coordinate(
    wavelengths_nm,
    coordinate_kind,
    material_dataset,
):
    """Convert requested wavelengths to a dataset coordinate."""
    if coordinate_kind == "wavelength_um":
        return wavelengths_nm / 1e3

    if coordinate_kind == "energy_ev":
        return _HC_EV_NM / wavelengths_nm

    raise ValueError(
        f"Unknown coordinate kind {coordinate_kind!r} for "
        f"{material_dataset!r}. Supported kinds are 'wavelength_um' "
        "and 'energy_ev'."
    )


def _validate_spectral_coverage(
    wavelengths_nm,
    optical_coordinates,
    coordinate_kind,
    material_dataset,
):
    """Raise a Warning when the optical data will be extrapolated."""
    requested_coordinates = _wavelength_nm_to_coordinate(
        wavelengths_nm,
        coordinate_kind=coordinate_kind,
        material_dataset=material_dataset,
    )
    requested_min = float(np.min(requested_coordinates))
    requested_max = float(np.max(requested_coordinates))
    available_min = float(optical_coordinates[0])
    available_max = float(optical_coordinates[-1])

    tolerance = (
        32
        * np.finfo(float).eps
        * max(
            1.0,
            abs(available_min),
            abs(available_max),
        )
    )

    if (
        requested_min < available_min - tolerance
        or requested_max > available_max + tolerance
    ):
        warnings.warn(
            "The requested wavelengths extend beyond the available optical data, "
            "which will be extrapolated."
        )


def _interpolate_permittivity(
    wavelengths_nm,
    optical_coordinates,
    first_optical_component,
    second_optical_component,
    coordinate_kind,
    representation,
    material_dataset,
):
    """Interpolate optical data and return the complex permittivity."""
    requested_coordinates = _wavelength_nm_to_coordinate(
        wavelengths_nm,
        coordinate_kind=coordinate_kind,
        material_dataset=material_dataset,
    )

    interpolated_first = np.interp(
        requested_coordinates,
        optical_coordinates,
        first_optical_component,
    )
    interpolated_second = np.interp(
        requested_coordinates,
        optical_coordinates,
        second_optical_component,
    )

    if representation == "n_k":
        return (interpolated_first + 1j * interpolated_second) ** 2

    if representation == "epsilon":
        return interpolated_first + 1j * interpolated_second

    raise ValueError(f"Unknown optical-data representation {representation!r}.")


def _electron_velocity_au(electron_energy_kev):
    """Return the relativistic electron velocity in atomic units."""
    kinetic_energy_au = electron_energy_kev * 1e3 / _AU_ENERGY_EV
    gamma = 1 + kinetic_energy_au / _C_AU**2
    beta = np.sqrt(1 - gamma**-2)

    return _C_AU * beta


def _calculate_transition_radiation(
    angles_deg,
    wavelengths_nm,
    electron_energy_kev,
    permittivity,
):
    """Calculate integrated and angle-resolved TR probability densities.

    Parameters
    ----------
    angles_deg : numpy.ndarray
        Strictly increasing polar angles in degrees.
    wavelengths_nm : numpy.ndarray
        Strictly increasing wavelengths in nm.
    electron_energy_kev : float
        Electron kinetic energy in keV.
    permittivity : numpy.ndarray
        Complex relative permittivity on ``wavelengths_nm``.

    Returns
    -------
    probability_nm : numpy.ndarray
        Angle-integrated emission-probability density per electron and nm.
    angular_probability_nm_sr : numpy.ndarray
        Emission-probability density per electron, nm, and steradian.
        Its shape is ``(number_of_angles, number_of_wavelengths)``.
    """
    photon_energy_ev = _HC_EV_NM / wavelengths_nm
    photon_energy_au = photon_energy_ev / _AU_ENERGY_EV

    theta_rad = np.deg2rad(angles_deg)[:, np.newaxis]
    omega = photon_energy_au[np.newaxis, :]
    epsilon = np.asarray(permittivity, dtype=complex)[np.newaxis, :]

    if epsilon.shape[1] != wavelengths_nm.size:
        raise ValueError("`permittivity` must contain one value for every wavelength.")

    electron_velocity = _electron_velocity_au(electron_energy_kev)
    wavevector_squared = (omega / _C_AU) ** 2

    parallel_wavevector = np.sin(theta_rad) * np.sqrt(wavevector_squared)
    electron_wavevector_squared = (
        parallel_wavevector**2 + (omega / electron_velocity) ** 2
    )

    vacuum_normal_wavevector = np.sqrt(wavevector_squared - parallel_wavevector**2 + 0j)
    material_normal_wavevector = np.sqrt(
        wavevector_squared * epsilon - parallel_wavevector**2
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        interface_factor = (
            2j
            * parallel_wavevector
            / _C_AU
            / (vacuum_normal_wavevector * epsilon + material_normal_wavevector)
        )

        source_factor = (
            material_normal_wavevector - (omega / electron_velocity) * epsilon
        ) / (wavevector_squared - electron_wavevector_squared) - (
            material_normal_wavevector - omega / electron_velocity
        ) / (wavevector_squared * epsilon - electron_wavevector_squared)

        far_field_amplitude = (
            1j
            * np.sqrt(wavevector_squared)
            * np.cos(theta_rad)
            * interface_factor
            * source_factor
        )

        far_field_squared = np.abs(far_field_amplitude) ** 2

        matching_medium = np.isclose(
            epsilon,
            1.0 + 0j,
            rtol=1e-13,
            atol=1e-15,
        )
        far_field_squared = np.where(
            matching_medium,
            0.0,
            far_field_squared,
        )

        angular_probability_ev_sr = (
            _C_AU / (4 * np.pi**2 * photon_energy_ev[np.newaxis, :]) * far_field_squared
        )

        energy_to_wavelength_jacobian = photon_energy_ev**2 / _HC_EV_NM
        angular_probability_nm_sr = (
            angular_probability_ev_sr * energy_to_wavelength_jacobian[np.newaxis, :]
        )

    if not np.all(np.isfinite(angular_probability_nm_sr)):
        raise FloatingPointError(
            "The transition-radiation calculation produced non-finite "
            "values. Check the selected material data and spectral range."
        )

    probability_nm = (
        2
        * np.pi
        * np.trapezoid(
            angular_probability_nm_sr * np.sin(theta_rad),
            x=theta_rad[:, 0],
            axis=0,
        )
    )

    probability_nm = np.maximum(probability_nm, 0.0)
    angular_probability_nm_sr = np.maximum(
        angular_probability_nm_sr,
        0.0,
    )

    return probability_nm, angular_probability_nm_sr
