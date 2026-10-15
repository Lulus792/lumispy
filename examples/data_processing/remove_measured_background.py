"""
Remove a measured dark background
=================================

This example shows how to remove a measured dark spectrum using the
``Scalable fixed pattern`` background model added in HyperSpy 2.5.0.
"""

# %%
# Load the experimental AlN spectrum and the measured dark background.
import matplotlib.pyplot as plt
import lumispy as lum

s = lum.data.AlN_Cornell()
bkg = lum.data.AlN_bkg()

# Average all 100 spectra to obtain a single spectrum for the background.
s = s.mean(axis=(0, 1))
bkg = bkg.mean(axis=(0, 1))

# %%
# Subtract the measured background without scaling it. ``signal_range='full'``
# enables non-interactive background removal over the complete spectrum.
# A different integration time or acquisition setting might require scaling.
corrected = s.remove_background(
    signal_range="full",
    background_type="Scalable fixed pattern",
    background_signal=bkg,
    yscale=False,
)

# %%
# Compare both measured spectra and the background-subtracted spectrum.
axis = s.axes_manager.signal_axes[0].axis
fig, ax = plt.subplots()
ax.plot(axis, s.data, label="Measured spectrum")
ax.plot(axis, bkg.data, label="Dark background")
ax.plot(axis, corrected.data, label="Background subtracted")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (a.u.)")
ax.legend()
plt.show()
