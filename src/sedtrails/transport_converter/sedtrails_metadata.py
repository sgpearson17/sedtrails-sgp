from dataclasses import dataclass
from typing import Any, Dict, Mapping

@dataclass
class SedtrailsMetadata:
    """
    A dataclass for storing metadata used by the SedtrailsData class.
    Domain / grid metadata for a SedtrailsData instance.

    The flowfield_domain is required and contains the spatial bounds of the domain.
    Additional metadata can be added as attributes dynamically.

    Attributes
    ----------
    flowfield_domain : dict
        Required dictionary containing:
        - x_min : float
        - x_max : float  
        - y_min : float
        - y_max : float
    """
    
    flowfield_domain: Dict[str, float]
    
    REQUIRED_DOMAIN_KEYS = {"x_min", "x_max", "y_min", "y_max"}
    RESERVED_KEYS = {"flowfield_domain"}

    def __post_init__(self):
        """Validate that flowfield_domain contains required keys."""
        if not isinstance(self.flowfield_domain, dict):
            raise TypeError("flowfield_domain must be a dictionary")
        
        missing_keys = self.REQUIRED_DOMAIN_KEYS - set(self.flowfield_domain.keys())
        if missing_keys:
            raise ValueError(f"flowfield_domain missing required keys: {missing_keys}")
        
        # Ensure all values are float
        for key in self.REQUIRED_DOMAIN_KEYS:
            self.flowfield_domain[key] = float(self.flowfield_domain[key])

    def __setattr__(self, name: str, value: Any):
        """Allow setting dynamic attributes while protecting reserved ones."""
        if name in self.RESERVED_KEYS and hasattr(self, name):
            # Only allow setting flowfield_domain during initialization
            if name == "flowfield_domain" and not hasattr(self, '_initialized'):
                super().__setattr__(name, value)
                super().__setattr__('_initialized', True)
            else:
                raise ValueError(f"'{name}' is reserved and cannot be overwritten")
        else:
            super().__setattr__(name, value)

    def add(self, key: str, value: Any):
        """Add a single metadata entry as an attribute."""
        setattr(self, key, value)

    def update(self, metadata_dict: Mapping[str, Any]):
        """Add multiple metadata entries as attributes."""
        for key, value in metadata_dict.items():
            setattr(self, key, value)

    def get(self, key: str, default=None) -> Any:
        """Get metadata value by key."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export all metadata as dictionary.
        
        Returns
        -------
        dict
            Complete metadata dictionary including dynamic attributes
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Custom repr showing all attributes."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"SedtrailsMetadata({attrs})"