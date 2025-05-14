"""
Flow Field Data Retriever

Retrieves flow field data from the transport converter or the simulation caching and state tracker.
Performs temporal interpolation to provide accurate flow field data at any time point for particle tracing.
"""
from typing import Tuple
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to allow importing from other packages
sys.path.append(str(Path(__file__).parent.parent.parent))
from sedtrails.transport_converter.format_converter import SedtrailsData


class FlowFieldDataRetriever:
    """
    Retrieves and interpolates flow field data from SedtrailsData.
    
    This class provides methods to extract flow fields at specific times,
    performing temporal interpolation as needed. Currently supports depth-averaged
    flow velocity.
    """
    
    def __init__(self, sedtrails_data: SedtrailsData):
        """
        Initialize the FlowFieldDataRetriever.
        
        Parameters:
        -----------
        sedtrails_data : SedtrailsData
            The SedtrailsData object containing the flow fields
        """
        self.sedtrails_data = sedtrails_data
        
        # Only using depth-averaged flow velocity for now
        self.flow_field_name = "depth_avg_flow_velocity"
    
    def get_interpolation_indices(self, target_time: float) -> Tuple[int, int, float]:
        """
        Get indices for interpolation between two time steps.
        
        Parameters:
        -----------
        target_time : float
            Target time in seconds since reference_date
            
        Returns:
        --------
        Tuple[int, int, float]
            (lower_index, upper_index, weight) where weight is the interpolation factor [0-1]
        """
        times = self.sedtrails_data.times
        
        # Handle edge cases
        if target_time <= times[0]:
            return 0, 0, 0.0
        
        if target_time >= times[-1]:
            last_idx = len(times) - 1
            return last_idx, last_idx, 1.0
        
        # Find the index of the last time that is less than or equal to the target time
        lower_idx = np.searchsorted(times, target_time, side='right') - 1
        upper_idx = lower_idx + 1
        
        # Calculate the interpolation weight
        time_range = times[upper_idx] - times[lower_idx]
        
        # Avoid division by zero
        if time_range == 0:
            weight = 0.0
        else:
            weight = (target_time - times[lower_idx]) / time_range
            
        return lower_idx, upper_idx, weight
    
    def get_flow_field(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the flow field at the specified time.
        
        This method performs temporal interpolation between the two closest time steps
        in the SedtrailsData object to obtain the flow field at the requested time.
        
        Parameters:
        -----------
        time : float
            Time in seconds since the reference date of the SedtrailsData object
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (flow_x, flow_y) arrays
        """
        # Get indices for interpolation
        lower_idx, upper_idx, weight = self.get_interpolation_indices(time)
        
        # If time is exactly at a time step or outside the range, no interpolation needed
        if lower_idx == upper_idx:
            time_slice = self.sedtrails_data[lower_idx]
            flow_field = time_slice[self.flow_field_name]
            return flow_field['x'], flow_field['y']
        
        # Otherwise, perform linear interpolation between the two time steps
        lower_slice = self.sedtrails_data[lower_idx]
        upper_slice = self.sedtrails_data[upper_idx]
        
        lower_flow = lower_slice[self.flow_field_name]
        upper_flow = upper_slice[self.flow_field_name]
        
        # Linear interpolation: result = (1-w)*lower + w*upper
        flow_x = (1 - weight) * lower_flow['x'] + weight * upper_flow['x']
        flow_y = (1 - weight) * lower_flow['y'] + weight * upper_flow['y']
        
        return flow_x, flow_y


if __name__ == "__main__":
    # Example usage - this requires SedtrailsData from format_converter.py
    try:
        # Import from the correct location
        from sedtrails.transport_converter.format_converter import FormatConverter, InputType
        
        # Default file path (can be overridden by command line argument)
        file_path = r"c:\Users\weste_bt\GitHub\sedtrails_data\inlet_sedtrails.nc"
        
        # Allow command line override of file path
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        
        # Create converter and read data
        converter = FormatConverter(file_path, input_type=InputType.NETCDF_DFM)
        converter.read_data()
        
        # Convert to SedtrailsData
        sedtrails_data = converter.convert_to_sedtrails_data()
        
        # Create flow field data retriever
        retriever = FlowFieldDataRetriever(sedtrails_data)
        
        # Choose a time for interpolation (30% between first and last time step)
        test_time = sedtrails_data.times[0] + (sedtrails_data.times[-1] - sedtrails_data.times[0]) * 0.3
        
        # Get flow field at the test time
        flow_x, flow_y = retriever.get_flow_field(test_time)
        
        # Print some information
        print(f"Time: {test_time} seconds since {sedtrails_data.reference_date}")
        print(f"Flow field shape - X: {flow_x.shape}, Y: {flow_y.shape}")
        
        # Calculate mean and max flow for verification
        mean_fx = np.nanmean(flow_x)
        max_fx = np.nanmax(np.abs(flow_x))
        mean_fy = np.nanmean(flow_y)
        max_fy = np.nanmax(np.abs(flow_y))
        
        print(f"Mean flow: ({mean_fx:.5f}, {mean_fy:.5f}) m/s")
        print(f"Max flow magnitude: ({max_fx:.5f}, {max_fy:.5f}) m/s")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the format_converter.py module is in the correct location:")
        print("src/sedtrails/transport_converter/format_converter.py")