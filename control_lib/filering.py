from collections import deque
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d

class LiDARPointTracker:
    def __init__(self, distance_threshold=0.5, vanish_threshold=3, smoothing_sigma=1, history_size=4):
        self.distance_threshold = distance_threshold  # Max distance to match points
        self.vanish_threshold = vanish_threshold  # Frames before a point is considered vanished
        self.smoothing_sigma = smoothing_sigma  # Smoothing parameter
        self.history_size = history_size  # Max number of frames to keep in history
        self.points = {}  # {point_id: {"position": [x, y], "history": deque, "age": n}}
        self.next_point_id = 0  # ID for new points

    def update(self, current_points):
        """
        Update the tracker with new points.
        """
        current_points = np.array(current_points)
        if len(current_points) == 0:
            # No new points detected, just update age of existing points
            self._age_existing_points()
            return

        # Build KDTree for existing points
        existing_positions = np.array([data["position"] for data in self.points.values()])
        tree = KDTree(existing_positions) if len(existing_positions) > 0 else None

        # Match new points to existing points
        matches = {}
        unmatched_points = set(range(len(current_points)))
        if tree is not None:
            distances, indices = tree.query(current_points, distance_upper_bound=self.distance_threshold)
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if idx < len(existing_positions) and dist < self.distance_threshold:
                    point_id = list(self.points.keys())[idx]
                    matches[point_id] = current_points[i]
                    unmatched_points.remove(i)

        # Update matched points
        for point_id, new_position in matches.items():
            self._update_point(point_id, new_position)

        # Add new points
        for i in unmatched_points:
            self._add_new_point(current_points[i])

        # Age points that were not matched
        self._age_existing_points(exclude=matches.keys())

    def _update_point(self, point_id, new_position):
        """
        Update the position and history of a matched point.
        """
        point_data = self.points[point_id]
        point_data["position"] = new_position
        point_data["history"][0].append(new_position[0])  # Update X history
        point_data["history"][1].append(new_position[1])  # Update Y history
        point_data["age"] = 0  # Reset age

    def _add_new_point(self, position):
        """
        Add a new point to the tracker.
        """
        self.points[self.next_point_id] = {
            "position": position,
            "history": [deque([position[0]], maxlen=self.history_size),  # X history
                        deque([position[1]], maxlen=self.history_size)],  # Y history
            "age": 0,
        }
        self.next_point_id += 1

    def _age_existing_points(self, exclude=set()):
        """
        Age points that were not matched and remove those exceeding the vanish threshold.
        """
        to_remove = []
        for point_id, data in self.points.items():
            if point_id in exclude:
                continue
            data["age"] += 1
            if data["age"] > self.vanish_threshold:
                to_remove.append(point_id)
        for point_id in to_remove:
            del self.points[point_id]

    def get_smoothed_points(self):
        """
        Return smoothed positions for all tracked points.
        """
        smoothed_positions = {}
        for point_id, data in self.points.items():
            smoothed_x = gaussian_filter1d(list(data["history"][0]), sigma=self.smoothing_sigma)
            smoothed_y = gaussian_filter1d(list(data["history"][1]), sigma=self.smoothing_sigma)
            smoothed_positions[point_id] = (smoothed_x[-1], smoothed_y[-1])  # Use the most recent smoothed position
        return smoothed_positions
