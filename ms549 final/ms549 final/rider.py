class Rider:
    def __init__(self, rider_id, start_location, destination):
        """
        Initialize a Rider object.
        
        Args:
            rider_id (str): A unique identifier for the rider (e.g., "RIDER_A")
            start_location: The rider's starting location (node ID)
            destination: The rider's final destination (node ID)
        """
        self.id = rider_id
        self.start_location = start_location
        self.destination = destination
        self.status = "waiting"
    
    def __str__(self):
        """
        Return a formatted string summarizing the rider's status.
        
        Returns:
            str: Formatted string with rider ID, location, and destination
        """
        return f"Rider {self.id} at {self.start_location} waiting for ride to {self.destination}"