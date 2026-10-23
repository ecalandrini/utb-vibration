import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from enum import StrEnum
import re
import pandas as pd
import numpy as np
from scipy import signal


class MeasurementType(StrEnum):
    SWEEP = "sweep"
    SHOCK = "shock"


class BaseMeasurement(ABC):
    """
    Generic base providing file validation and text normalization utilities.
    Does not make assumptions about sheets, units, or column names.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path.resolve()}")
        self.df: pd.DataFrame = self._parse()

    @abstractmethod
    def _parse(self) -> pd.DataFrame:
        """Each instrument subclass implements its own parsing pipeline."""
        pass

    @staticmethod
    def clean_header(col: str) -> str:
        """Sanitize messy hardware column labels to snake_case."""
        return re.sub(r"[\[\(\]\)\/\s]+", "_", str(col)).strip("_").lower()

    @staticmethod
    def parse_comma_decimals(df: pd.DataFrame) -> pd.DataFrame:
        """Convert European comma decimal strings to float64."""
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
                except ValueError:
                    pass
        return df


class Daq(BaseMeasurement):
    """
    DAQ logger: Exports multi-level headers across multiple sheets containing a specific string.
    Combines them sequentially.
    """

    def __init__(self, file_path: str | Path, sheet_pattern: str = "Data1"):
        self.sheet_pattern = sheet_pattern
        super().__init__(file_path)

    def _parse(self) -> pd.DataFrame:
        # 1. Open the Excel file metadata without fully loading all sheet data
        xls = pd.ExcelFile(self.file_path)

        # 2. Identify target sheets maintaining the original workbook order
        target_sheets = [
            sheet for sheet in xls.sheet_names if self.sheet_pattern in sheet
        ]

        if not target_sheets:
            raise ValueError(
                f"No sheets matching '{self.sheet_pattern}' found in {self.file_path}"
            )

        # 3. Read all target sheets at once (returns a dict: {sheet_name: DataFrame})
        dfs_dict = pd.read_excel(xls, sheet_name=target_sheets, header=[0, 1])

        processed_dfs = []
        for sheet_name, df in dfs_dict.items():
            # Flatten and sanitize headers for this specific sheet
            df.columns = [
                (
                    self.clean_header(f"{col[0]}_{col[1]}")
                    if str(col[1]) != "nan" and not str(col[1]).startswith("Unnamed")
                    else self.clean_header(col[0])
                )
                for col in df.columns
            ]
            processed_dfs.append(df)

        # 4. Combine all sheets vertically. ignore_index=True resets row numbers 0 to N.
        combined_df = pd.concat(processed_dfs, ignore_index=True, sort=False)

        # 5. Clean decimal commas globally on the combined dataframe
        combined_df = self.parse_comma_decimals(combined_df)
        print(combined_df.columns)

        sort_column = "time_s"

        if sort_column in combined_df.columns:
            # Sort ascending (oldest to newest) and reset the index
            combined_df = combined_df.sort_values(
                by=sort_column, ascending=True, ignore_index=True
            )
        else:
            # Raise an error or log a warning if the expected time column is missing
            raise KeyError(
                f"Cannot sort: '{sort_column}' not found in columns: {list(combined_df.columns)}"
            )

        return combined_df


class Shaker(BaseMeasurement):
    """
    Shaker controller: Exports single-row CSVs with time in milliseconds.
    """

    def __init__(self, file_path: str | Path, measurement_type: str | MeasurementType):
        self.measurement_type = MeasurementType(measurement_type.lower())
        super().__init__(file_path)

    def _parse(self) -> pd.DataFrame:
        # 1. Read CSV
        csv_kwargs = {
            MeasurementType.SWEEP: {},
            MeasurementType.SHOCK: {"delimiter": ";"},
        }
        kwargs = csv_kwargs.get(self.measurement_type)

        if kwargs is None:
            raise NotImplementedError(
                f"Parsing not defined for {self.measurement_type}"
            )

        # Unpack the specific arguments into read_csv
        df = pd.read_csv(self.file_path, **kwargs)

        # 2. Sanitize column headers
        df = df.rename(columns=self.clean_header)

        # 3. Instrument-specific unit normalization
        if "time_ms" in df.columns:
            df["time_s"] = df.pop("time_ms") / 1000.0

        return df


folder = Path(r"C:\Users\ecalandrini\Projects\UTB_App\utb-vibration\data")
file_shaker = "T3_SINE_X_01.csv"
file_daq = "Test SENO 5 khZ ii.xlsx"
shaker = Shaker(folder / file_shaker, measurement_type="Shock")
daq = Daq(folder / file_daq)

print(shaker.df.columns)
print(daq.df.columns)

t_min = 6
t_max = 271
df = daq.df.query("@t_min < time_s < @t_max")

fig, ax = plt.subplots()
# shaker.df.plot("time_s", "monitor_c1_m_s2", ax=ax)
# mask = daq.df["time_s"].between(2, 4, inclusive="both")
df.plot("time_s", "ai_1_m_s2", ax=ax)
plt.show()

# Assuming your DataFrame and signal are already loaded:
time_col = df["time_s"].values
signal_col = df["ai_1_m_s2"].values
fs = 5000

# 1. Spectrogram (STFT) Configuration
# nperseg=5000 gives exactly 1 Hz frequency resolution
nperseg = 5000
f, t, Sxx = signal.spectrogram(
    signal_col,
    fs=fs,
    window="hann",
    nperseg=nperseg,
    noverlap=nperseg // 20,
    # nfft=2 * nperseg,
    scaling="spectrum",
    detrend="constant",
)

# Sxx[Sxx < 2] = 1

# 2. Extract 1D Max-Hold Amplitude
# Take the maximum power across the time axis (axis=1)
max_power_spectrum = np.max(Sxx, axis=1)
# Convert power back to linear peak amplitude
amplitude_1d = np.sqrt(max_power_spectrum * 2)


# 3. Visualization Setup
fig, (ax1, ax2) = plt.subplots(2, 1)

# --- Plot 1: 2D Spectrogram ---
# Convert Sxx to dB for visual clarity in the colormap
pcm = ax1.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
ax1.set_title("2D Spectrogram: System Response to Sine Sweep")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_xlabel("Time (sec)")
ax1.set_ylim(0, 250)  # Constrained to your relevant sweep area
fig.colorbar(pcm, ax=ax1, label="Power Spectrum Magnitude (dB)")

# --- Plot 2: 1D Max-Hold Envelope ---
ax2.plot(f, amplitude_1d, color="red", linewidth=2)
ax2.set_title("1D Max-Hold Envelope (Extracted Peak Amplitude)")
ax2.set_ylabel("Peak Amplitude (m/s² or g)")
ax2.set_xlabel("Frequency (Hz)")
ax2.grid(True, which="both")
ax2.set_xlim(0, 200)  # Constrained to the physical sweep limits

plt.tight_layout()
plt.show()
# Define the index you want to inspect
# time_index = 5100000

# # Extract the physical time in seconds for your plot title
# physical_time = t[time_index]

# # Extract the vertical cut (Power Spectrum)
# slice_power = Sxx[:, time_index]

# # Convert Power to Peak Linear Amplitude (g)
# slice_amplitude = np.sqrt(slice_power * 2)

# # Convert to dB (optional, depending on your preferred visualization)
# # slice_dB = 20 * np.log10(slice_amplitude)

# plt.figure()
# plt.plot(f, slice_amplitude)  # or plot slice_dB
# plt.title(f"Instantaneous Spectrum at t = {physical_time:.2f} seconds")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Peak Amplitude (g)")
# plt.grid(True, which="both")
# plt.xlim(0, 250)  # Constrain to your sweep area
# plt.show()

plt.figure()
plt.plot(
    daq.df["time_s"],
    daq.df["ai_1_m_s2"],
)
plt.show()

from scipy.fft import rfft, rfftfreq

# Assuming your signal is loaded
n_samples = len(signal_col)
fs = 5000.0  # Sampling frequency

# 1. Compute the one-dimensional discrete Fourier Transform for real input
yf = rfft(signal_col)

# 2. Compute the corresponding frequency bins
xf = rfftfreq(n_samples, 1 / fs)

# 3. Calculate peak amplitude magnitude
# Normalizing by (n_samples / 2) converts the FFT complex output to peak amplitude (g or m/s^2)
magnitude = np.abs(yf) / (n_samples / 2)

# Plotting the global FFT
plt.figure(figsize=(10, 5))
plt.plot(xf, magnitude)
plt.title("Global FFT of Swept Sine Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak Amplitude")
plt.grid(True, which="both")
plt.xlim(0, 250)  # Constrain to your sweep range + buffer
plt.show()

plt.figure(figsize=(10, 5))
step_size = 50000
delta_t = 2500
yy = 0
for t in range(0, len(daq.df), delta_t):
    yy += 0.1
    df = daq.df.loc[t : t + step_size]

    time_col = df["time_s"].values
    signal_col = df["ai_1_m_s2"].values

    n_samples = len(signal_col)
    fs = 5000.0  # Sampling frequency

    # 1. Compute the one-dimensional discrete Fourier Transform for real input
    yf = rfft(signal_col)

    # 2. Compute the corresponding frequency bins
    xf = rfftfreq(n_samples, 1 / fs)

    # 3. Calculate peak amplitude magnitude
    # Normalizing by (n_samples / 2) converts the FFT complex output to peak amplitude (g or m/s^2)
    magnitude = np.abs(yf) / (n_samples / 2)
    plt.plot(xf, magnitude)
plt.title("Global FFT of Swept Sine Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak Amplitude")
plt.grid(True, which="both")
plt.xlim(0, 250)  # Constrain to your sweep range + buffer
plt.show()
