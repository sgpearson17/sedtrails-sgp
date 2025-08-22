import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class SedtrailsData:
    """
    A data class for internally structuring SedTrails data.

    This class holds data for multiple time steps with time as the first dimension
    for time-dependent variables.
    """

    times: np.ndarray
    reference_date: np.datetime64
    x: np.ndarray
    y: np.ndarray
    bed_level: np.ndarray
    depth_avg_flow_velocity: Dict[str, np.ndarray]
    fractions: int
    bed_load_transport: Dict[str, np.ndarray]
    suspended_transport: Dict[str, np.ndarray]
    water_depth: np.ndarray
    mean_bed_shear_stress: np.ndarray
    max_bed_shear_stress: np.ndarray
    sediment_concentration: np.ndarray
    nonlinear_wave_velocity: Dict[str, np.ndarray]
    min_resolution: float
    outer_envelope: np.ndarray

    def __post_init__(self):
        """Initialize container for dynamic physics fields."""
        self._physics_fields: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Physics field management
    # ------------------------------------------------------------------
    def add_physics_field(self, name: str, data):
        """
        Add a physics field to the data structure.

        Parameters
        ----------
        name : str
            Name of the physics field
        data : np.ndarray or dict
            Physics data (scalar array or dict with 'x', 'y', 'magnitude' for vectors)
        """
        self._physics_fields[name] = data
        setattr(self, name, data)

    def has_physics_field(self, name: str) -> bool:
        """Check if a specific physics field exists."""
        return name in self._physics_fields

    def get_physics_fields(self) -> list:
        """Get list of available physics field names."""
        return list(self._physics_fields.keys())

    def has_physics_data(self) -> bool:
        """Check if any physics fields have been added."""
        return len(self._physics_fields) > 0

    # ------------------------------------------------------------------
    # Time slicing
    # ------------------------------------------------------------------
    def __getitem__(self, time_index: int) -> Dict:
        """
        Get data for a specific time index.

        Parameters
        ----------
        time_index : int
            Time index to extract

        Returns
        -------
        Dict
            Dictionary containing all data for the specified time index
        """
        if time_index < 0 or time_index >= len(self.times):
            raise IndexError(f"Time index {time_index} out of bounds (0-{len(self.times) - 1})")

        # Core fields
        data = {
            "time": self.times[time_index],
            "reference_date": self.reference_date,
            "x": self.x,
            "y": self.y,
            "bed_level": self.bed_level,  # typically time-independent
            "fractions": self.fractions,
            "min_resolution": self.min_resolution,
            "outer_envelope": self.outer_envelope,
            "water_depth": self.water_depth[time_index],
            "mean_bed_shear_stress": self.mean_bed_shear_stress[time_index],
            "max_bed_shear_stress": self.max_bed_shear_stress[time_index],
            "sediment_concentration": self.sediment_concentration[time_index],
            "depth_avg_flow_velocity": {
                "x": self.depth_avg_flow_velocity["x"][time_index],
                "y": self.depth_avg_flow_velocity["y"][time_index],
                "magnitude": self.depth_avg_flow_velocity["magnitude"][time_index],
            },
            "bed_load_transport": {
                "x": self.bed_load_transport["x"][time_index],
                "y": self.bed_load_transport["y"][time_index],
                "magnitude": self.bed_load_transport["magnitude"][time_index],
            },
            "suspended_transport": {
                "x": self.suspended_transport["x"][time_index],
                "y": self.suspended_transport["y"][time_index],
                "magnitude": self.suspended_transport["magnitude"][time_index],
            },
            "nonlinear_wave_velocity": {
                "x": self.nonlinear_wave_velocity["x"][time_index],
                "y": self.nonlinear_wave_velocity["y"][time_index],
                "magnitude": self.nonlinear_wave_velocity["magnitude"][time_index],
            },
        }

        # Dynamic physics fields
        for name, value in self._physics_fields.items():
            if isinstance(value, dict):  # vector field
                data[name] = {
                    "x": value["x"][time_index],
                    "y": value["y"][time_index],
                    "magnitude": value["magnitude"][time_index],
                }
            else:  # scalar field
                data[name] = value[time_index]

        return data
