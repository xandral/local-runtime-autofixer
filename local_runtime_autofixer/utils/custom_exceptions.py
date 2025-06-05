from local_runtime_autofixer.utils.models import Incident
from typing import Optional


class IncidentException(Exception):
    """Generic Exception for this project"""

    def __init__(self, message: str, incident: Optional[Incident] = None):
        self.message = message
        self.incident = incident

        super().__init__(message)

    def get_analysis_details(self):
        """Return all security analysis details for use in auto_fix."""
        return self.incident
