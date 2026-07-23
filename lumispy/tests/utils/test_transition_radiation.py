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

import numpy as np
from numpy.testing import assert_allclose
import pytest

from hyperspy.axes import DataAxis
from hyperspy.signals import Signal1D, Signal2D

from lumispy.utils.transition_radiation import (
    _DATASETS,
    _HC_EV_NM,
    _interpolate_permittivity,
    _load_optical_data,
    transition_radiation_nm,
)


def test_transition_radiation_returns_calibrated_signals():
    wavelengths_nm = np.array([400.0, 550.0, 800.0])
    angles_deg = np.array([0.0, 30.0, 60.0, 90.0])

    spectrum, angle_resolved = transition_radiation_nm(
        electron_energy=30,
        material_dataset="Al_mcpeak",
        angular_axis=angles_deg,
        spectral_axis=wavelengths_nm,
        return_angle_resolved=True,
    )

    assert isinstance(spectrum, Signal1D)
    assert isinstance(angle_resolved, Signal2D)
    assert spectrum.data.shape == (wavelengths_nm.size,)
    assert angle_resolved.data.shape == (angles_deg.size, wavelengths_nm.size)

    spectrum_axis = spectrum.axes_manager.signal_axes[0]
    angle_resolved_axes = {
        axis.name: axis for axis in angle_resolved.axes_manager.signal_axes
    }
    assert spectrum_axis.name == "Wavelength"
    assert spectrum_axis.units == "nm"
    assert_allclose(spectrum_axis.axis, wavelengths_nm)
    assert angle_resolved_axes["Angle"].units == "deg"
    assert_allclose(angle_resolved_axes["Angle"].axis, angles_deg)
    assert_allclose(
        angle_resolved_axes["Wavelength"].axis,
        wavelengths_nm,
    )

    assert spectrum.metadata.Transition_radiation.electron_energy_keV == 30
    assert angle_resolved.metadata.Transition_radiation.material_dataset == "Al_mcpeak"
    assert np.all(np.isfinite(spectrum.data))
    assert np.all(np.isfinite(angle_resolved.data))
    assert np.all(spectrum.data >= 0)
    assert np.all(angle_resolved.data >= 0)


def test_angle_resolved_probability_integrates_to_spectrum():
    angles_deg = np.linspace(0, 90, 721)
    spectrum, angle_resolved = transition_radiation_nm(
        electron_energy=30,
        material_dataset="Al_mcpeak",
        angular_axis=angles_deg,
        spectral_axis=[400.0, 600.0, 800.0],
        return_angle_resolved=True,
    )

    angles_rad = np.deg2rad(angles_deg)
    integrated = (
        2
        * np.pi
        * np.trapezoid(
            angle_resolved.data * np.sin(angles_rad)[:, np.newaxis],
            x=angles_rad,
            axis=0,
        )
    )

    assert_allclose(integrated, spectrum.data, rtol=1e-13, atol=0)


@pytest.mark.parametrize("material_dataset", sorted(_DATASETS))
def test_all_bundled_optical_datasets(material_dataset):
    spectrum = transition_radiation_nm(
        material_dataset=material_dataset,
        angular_axis=(0, 90, 181),
        spectral_axis=[400.0, 600.0, 800.0],
    )

    assert spectrum.data.shape == (3,)
    assert np.all(np.isfinite(spectrum.data))
    assert np.all(spectrum.data >= 0)


def test_palik_data_are_used_as_permittivity():
    (
        optical_energy_ev,
        coordinate_kind,
        representation,
        real_permittivity,
        imaginary_permittivity,
    ) = _load_optical_data("Au_palik")
    index = optical_energy_ev.size // 2
    wavelength_nm = _HC_EV_NM / optical_energy_ev[index : index + 1]

    interpolated = _interpolate_permittivity(
        wavelength_nm,
        optical_energy_ev,
        real_permittivity,
        imaginary_permittivity,
        coordinate_kind=coordinate_kind,
        representation=representation,
        material_dataset="Au_palik",
    )

    expected = real_permittivity[index] + 1j * imaginary_permittivity[index]
    assert representation == "epsilon"
    assert_allclose(interpolated, expected, rtol=1e-13, atol=0)
    assert not np.isclose(interpolated[0], expected**2)


def test_n_and_k_data_are_converted_to_permittivity():
    (
        optical_wavelength_um,
        coordinate_kind,
        representation,
        refractive_index,
        extinction_coefficient,
    ) = _load_optical_data("Au_johnson")
    index = optical_wavelength_um.size // 2
    wavelength_nm = optical_wavelength_um[index : index + 1] * 1e3

    interpolated = _interpolate_permittivity(
        wavelength_nm,
        optical_wavelength_um,
        refractive_index,
        extinction_coefficient,
        coordinate_kind=coordinate_kind,
        representation=representation,
        material_dataset="Au_johnson",
    )

    expected = (refractive_index[index] + 1j * extinction_coefficient[index]) ** 2
    assert representation == "n_k"
    assert_allclose(interpolated, expected, rtol=1e-13, atol=0)


def test_hyperspy_axes_are_accepted():
    angular_axis = DataAxis(axis=np.linspace(0, 90, 91))
    spectral_axis = DataAxis(axis=np.array([400.0, 600.0, 800.0]))

    spectrum = transition_radiation_nm(
        angular_axis=angular_axis,
        spectral_axis=spectral_axis,
    )

    assert_allclose(
        spectrum.axes_manager.signal_axes[0].axis,
        spectral_axis.axis,
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (
            {"angular_axis": (90, 0)},
            ValueError,
            "strictly increasing",
        ),
        (
            {"material_dataset": "not-a-dataset"},
            ValueError,
            "Unknown material dataset",
        ),
    ],
)
def test_invalid_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        transition_radiation_nm(**kwargs)


def test_spectral_extrapolation_warns():
    with pytest.warns(UserWarning, match="beyond the available optical data"):
        spectrum = transition_radiation_nm(
            material_dataset="Au_sc",
            spectral_axis=(150, 1200),
        )

    assert spectrum.data.shape == (200,)
    assert np.all(np.isfinite(spectrum.data))
