#!/usr/bin/env python3
"""
Person Tracking Module for Safety Detection System

This module implements centroid-based tracking to prevent counting the same person
multiple times across video frames. It uses distance-based association and handles
person entry/exit scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import math

class PersonTracker:
    """
    Centroid-based person tracker for safety monitoring
    
    This tracker maintains unique IDs for detected persons across frames,
    preventing duplicate counting and enabling temporal safety analysis.
    """
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 max_distance: float = 100.0):
        """
        Initialize the person tracker
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removal
            max_distance: Maximum distance for person association (pixels)
        """
        # Tracking state
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> centroid
        self.disappeared = OrderedDict()  # object_id -> frames_disappeared
        
        # Configuration
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Safety tracking per person
        self.person_safety_status = OrderedDict()  # object_id -> safety_data
        
    def register(self, centroid: Tuple[float, float]) -> int:
        """
        Register a new person with a unique ID
        
        Args:
            centroid: (x, y) center point of the person
            
        Returns:
            Unique person ID
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.person_safety_status[self.next_object_id] = {
            'has_helmet': False,
            'has_jacket': False,
            'last_seen_frame': 0,
            'total_frames': 0,
            'violation_frames': 0
        }
        
        person_id = self.next_object_id
        self.next_object_id += 1
        
        return person_id
    
    def deregister(self, object_id: int):
        """
        Remove a person from tracking
        
        Args:
            object_id: ID of person to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.person_safety_status:
            del self.person_safety_status[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with person info
            
        Returns:
            Dictionary mapping person_id to detection data with safety status
        """
        # Extract person detections and their centroids
        person_detections = [d for d in detections if d['equipment_type'] == 'person']
        
        if len(person_detections) == 0:
            # No persons detected, mark all as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove persons that have been gone too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return {}
        
        # Get centroids from current detections
        input_centroids = np.array([[d['center'][0], d['center'][1]] for d in person_detections])
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            tracked_persons = {}
            for i, detection in enumerate(person_detections):
                person_id = self.register(input_centroids[i])
                tracked_persons[person_id] = self._update_person_safety(
                    person_id, detection, detections
                )
            return tracked_persons
        
        # Get existing object centroids
        object_centroids = np.array(list(self.objects.values()))
        
        # Compute distance matrix between existing objects and input centroids
        D = self._compute_distance_matrix(object_centroids, input_centroids)
        
        # Perform assignment using Hungarian algorithm (simplified greedy approach)
        object_ids = list(self.objects.keys())
        used_rows = set()
        used_cols = set()
        tracked_persons = {}
        
        # Find the minimum distance assignments
        for _ in range(min(len(object_ids), len(input_centroids))):
            # Find minimum distance pair not yet used
            min_dist = float('inf')
            min_row, min_col = -1, -1
            
            for row in range(D.shape[0]):
                if row in used_rows:
                    continue
                for col in range(D.shape[1]):
                    if col in used_cols:
                        continue
                    if D[row, col] < min_dist:
                        min_dist = D[row, col]
                        min_row, min_col = row, col
            
            # If minimum distance is acceptable, make assignment
            if min_dist <= self.max_distance:
                object_id = object_ids[min_row]
                
                # Update object centroid
                self.objects[object_id] = input_centroids[min_col]
                self.disappeared[object_id] = 0
                
                # Update safety tracking for this person
                tracked_persons[object_id] = self._update_person_safety(
                    object_id, person_detections[min_col], detections
                )
                
                used_rows.add(min_row)
                used_cols.add(min_col)
            else:
                break
        
        # Handle unmatched existing objects (mark as disappeared)
        for i, object_id in enumerate(object_ids):
            if i not in used_rows:
                self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        # Handle unmatched input centroids (register as new objects)
        for j in range(len(input_centroids)):
            if j not in used_cols:
                person_id = self.register(input_centroids[j])
                tracked_persons[person_id] = self._update_person_safety(
                    person_id, person_detections[j], detections
                )
        
        return tracked_persons
    
    def _compute_distance_matrix(self, 
                                object_centroids: np.ndarray, 
                                input_centroids: np.ndarray) -> np.ndarray:
        """
        Compute distance matrix between existing and new centroids
        
        Args:
            object_centroids: Existing object centroids (N x 2)
            input_centroids: New detection centroids (M x 2)
            
        Returns:
            Distance matrix (N x M)
        """
        # Compute Euclidean distance matrix
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
        return D
    
    def _update_person_safety(self, 
                             person_id: int, 
                             person_detection: Dict, 
                             all_detections: List[Dict]) -> Dict:
        """
        Update safety status for a tracked person
        
        Args:
            person_id: Unique person ID
            person_detection: Detection data for this person
            all_detections: All detections in current frame
            
        Returns:
            Updated person data with safety status
        """
        # Get safety equipment near this person
        person_bbox = person_detection['bbox']
        person_center = person_detection['center']
        
        # Check for helmet
        has_helmet = False
        for detection in all_detections:
            if detection['equipment_type'] == 'helmet':
                if self._is_equipment_associated(person_bbox, person_center, detection):
                    has_helmet = True
                    break
        
        # Check for jacket
        has_jacket = False
        for detection in all_detections:
            if detection['equipment_type'] == 'reflective_jacket':
                if self._is_equipment_associated(person_bbox, person_center, detection):
                    has_jacket = True
                    break
        
        # Update person safety tracking
        safety_data = self.person_safety_status[person_id]
        safety_data['has_helmet'] = has_helmet
        safety_data['has_jacket'] = has_jacket
        safety_data['total_frames'] += 1
        
        # Count violation frames
        is_compliant = has_helmet and has_jacket
        if not is_compliant:
            safety_data['violation_frames'] += 1
        
        # Create enriched person data
        person_data = person_detection.copy()
        person_data['person_id'] = person_id
        person_data['safety_status'] = {
            'has_helmet': has_helmet,
            'has_jacket': has_jacket,
            'is_compliant': is_compliant,
            'compliance_rate': ((safety_data['total_frames'] - safety_data['violation_frames']) 
                               / safety_data['total_frames'] * 100) if safety_data['total_frames'] > 0 else 100.0
        }
        
        return person_data
    
    def _is_equipment_associated(self, 
                                person_bbox: List[float], 
                                person_center: List[float], 
                                equipment_detection: Dict) -> bool:
        """
        Check if equipment is associated with a person
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            person_center: Person center point [x, y]
            equipment_detection: Equipment detection data
            
        Returns:
            True if equipment belongs to this person
        """
        equipment_bbox = equipment_detection['bbox']
        equipment_center = equipment_detection['center']
        
        # Check for bounding box overlap
        if self._check_bbox_overlap(person_bbox, equipment_bbox):
            return True
        
        # Check distance from person center
        distance = self._calculate_distance(person_center, equipment_center)
        
        # Different thresholds for different equipment types
        if equipment_detection['equipment_type'] == 'helmet':
            return distance < 80  # Helmets should be close to head
        elif equipment_detection['equipment_type'] == 'reflective_jacket':
            return distance < 120  # Jackets can be further from center
        
        return False
    
    def _check_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Check if bboxes don't overlap
        if (x1_max < x2_min or x2_max < x1_min or 
            y1_max < y2_min or y2_max < y1_min):
            return False
        
        return True
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_tracking_summary(self) -> Dict:
        """
        Get summary of current tracking state
        
        Returns:
            Dictionary with tracking statistics
        """
        total_persons = len(self.objects)
        active_persons = sum(1 for disappeared in self.disappeared.values() if disappeared == 0)
        
        # Calculate compliance statistics
        compliant_persons = 0
        total_violations = 0
        
        for person_id, safety_data in self.person_safety_status.items():
            if person_id in self.objects:  # Only count active persons
                if safety_data['has_helmet'] and safety_data['has_jacket']:
                    compliant_persons += 1
                total_violations += safety_data['violation_frames']
        
        return {
            'total_tracked_persons': total_persons,
            'active_persons': active_persons,
            'compliant_persons': compliant_persons,
            'compliance_rate': (compliant_persons / active_persons * 100) if active_persons > 0 else 100.0,
            'total_violation_frames': total_violations,
            'next_person_id': self.next_object_id
        }
    
    def reset(self):
        """Reset the tracker to initial state"""
        self.objects.clear()
        self.disappeared.clear()
        self.person_safety_status.clear()
        self.next_object_id = 0
