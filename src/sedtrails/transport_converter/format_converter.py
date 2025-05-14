"""
Format Converter: manage the conversion of input data formats and into a internal format.
"""

from dataclasses import dataclass, field


@dataclass
class SedtrailsData:
    """
    A data class for internal structuring SedTrails data.

    Attributes:
    -----------
        time: int
    """

    time: int  # current time in seconds
    x: int  # spatial coordinate in x direction
    y: int  # spatial coordinate in y direction
    time_step: int  # time step in seconds
    bed_level: float  # bed level in meters
    depth_avg_flow_velocity: float  # depth average flow velocity in m/s
    fractions: int  # number of fractions
    bed_load_sediment: float  # bed load sediment transport in kg/m/s
    suspended_sediment: float  # suspended sediment transport in kg/m/s
    water_depth: float  # water depth in meters
    mean_bed_shear_stress: float  # mean bed shear stress in pascal
    max_bed_shear_stress: float  # max bed shear stress in pascal
    sediment_concentration: float  # suspended sediment concentration in kg/m^3
    nonlinear_wave_velocity: float  # nonlinear wave velocity in m/s
    _previous = field(default=None, init=False)  # previous data in the sequence
    _next = field(default=None, init=False)  # next data in the sequence

    @property
    def previous(self):
        return self._previous

    @property
    def next(self):
        return self._next

    @previous.setter
    def previous(self, data_sequence):
        self._previous = data_sequence

    @next.setter
    def next(self, data_sequence):
        self._next = data_sequence

    def data_instance(self):
        """
        Returns an instance of the current data.
        """
        return self


if __name__ == '__main__':
    data = SedtrailsData(
        time=0,
        x=0,
        y=0,
        time_step=0,
        bed_level=0.0,
        depth_avg_flow_velocity=0.0,
        fractions=0,
        bed_load_sediment=0.0,
        suspended_sediment=0.0,
        water_depth=0.0,
        mean_bed_shear_stress=0.0,
        max_bed_shear_stress=0.0,
        sediment_concentration=0.0,
        nonlinear_wave_velocity=0.0,
    )
    print(data)

    data.previous = 'previous data'
    print(data.previous)

    # print(data.data_instance())
