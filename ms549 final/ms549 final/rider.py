class Rider:
    def __init__(self, rider_id, start_location, destination):
        self.id = rider_id
        self.start_location = start_location
        self.destination = destination
        self.status = 'waiting_for_car'  # waiting_for_car, waiting_for_pickup, in_car, completed
        self.assigned_car = None  # NEW: Link to assigned Car object
        self.request_time = None  # NEW: Will be set when request is made
        self.coordinates = None

    def __str__(self):
        """
        Return a formatted string summarizing the rider's status.
        
        Returns:
            str: Formatted string with rider ID, location, and destination
        """
        return f"Rider {self.id} at {self.start_location} waiting for ride to {self.destination}"