import os
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class DonkeycarDataLoader:
    def __init__(self):
        self.records = []
        self.base_path = None
        self.data_path = None  # Actual path where catalog files are located
        self.manifest = None
        self.sessions = set()
        self.data_keys = set()  # For tracking unique keys during loading
        self.ordered_data_keys = []  # Preserves catalog order
        self.deleted_indexes = []
        self.catalog_manifest = None
        
    def load_data(self, folder_path):
        """Load all catalog files from the specified folder"""
        self.base_path = folder_path
        self.records = []
        self.sessions = set()
        self.data_keys = set()
        self.ordered_data_keys = []
        self.deleted_indexes = []
        self.catalog_manifest = None

        # Check if catalog files exist directly in folder or in 'data' subfolder
        # Pattern 1: folder/data/*.catalog (Donkeycar format)
        # Pattern 2: folder/*.catalog (direct data folder)
        data_path = os.path.join(folder_path, 'data')
        if not os.path.exists(data_path) or not any(f.endswith('.catalog') for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))):
            # Check if catalog files exist directly in folder_path
            if any(f.endswith('.catalog') for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))):
                data_path = folder_path

        # Store the actual data path for image loading
        self.data_path = data_path

        # Load manifest
        manifest_path = os.path.join(data_path, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 5:
                    self.manifest = {
                        'keys': json.loads(lines[0]),
                        'types': json.loads(lines[1]),
                        'metadata': json.loads(lines[3]) if len(lines) > 3 else {}
                    }
                    # Load catalog manifest from line 5 (index 4)
                    if len(lines) > 4:
                        self.catalog_manifest = json.loads(lines[4])
                        self.deleted_indexes = self.catalog_manifest.get('deleted_indexes', [])
        
        # Load all catalog files in order
        catalog_files = sorted(glob.glob(os.path.join(data_path, 'catalog_*.catalog')))
        
        # Get all records with their original absolute indexes
        all_records = []
        for catalog_file in catalog_files:
            # Extract catalog number from filename
            catalog_name = os.path.basename(catalog_file)
            catalog_num = int(catalog_name.split('_')[1].split('.')[0])
            
            records_in_catalog = self._load_catalog_file(catalog_file, catalog_num)
            all_records.extend(records_in_catalog)
        
        # Sort by absolute index to maintain original order
        all_records.sort(key=lambda x: x.get('_absolute_index', 0))
        
        # Reassign continuous indexes and mark deleted
        for new_idx, record in enumerate(all_records):
            record['_display_index'] = new_idx
            if record.get('_absolute_index') in self.deleted_indexes:
                record['_is_deleted'] = True
            else:
                record['_is_deleted'] = False
        
        self.records = all_records

        # Extract ordered keys from first record to preserve catalog order
        if self.records:
            self.ordered_data_keys = list(self.records[0].keys())

    def _load_catalog_file(self, catalog_path, catalog_num):
        """Load records from a single catalog file"""
        records = []
        max_len = 1000  # Default max length per catalog
        
        # Get max_len from catalog manifest if available
        if self.catalog_manifest:
            max_len = self.catalog_manifest.get('max_len', 1000)
        
        with open(catalog_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        
                        # Calculate absolute index across all catalogs
                        local_index = record.get('_index', 0)
                        record['_absolute_index'] = catalog_num * max_len + local_index
                        
                        records.append(record)
                        
                        # Track sessions
                        if '_session_id' in record:
                            self.sessions.add(record['_session_id'])
                        
                        # Track data keys
                        self.data_keys.update(record.keys())
                    except json.JSONDecodeError:
                        continue
        
        return records
    
    def get_records(self, start_idx=0, end_idx=None, session_id=None):
        """Get records with optional filtering"""
        filtered_records = self.records
        
        # Filter by session
        if session_id:
            filtered_records = [r for r in filtered_records if r.get('_session_id') == session_id]
        
        # Apply pagination
        if end_idx is None:
            end_idx = len(filtered_records)
        
        return filtered_records[start_idx:end_idx]
    
    def get_sessions(self):
        """Get list of available sessions"""
        return sorted(list(self.sessions))
    
    def get_data_keys(self):
        """Get list of all data keys (preserves catalog order)"""
        return self.ordered_data_keys.copy()
    
    def get_numerical_keys(self):
        """Get list of numerical data keys (preserves catalog order)"""
        numerical_keys = []

        # Use ordered_data_keys to preserve catalog order
        for key in self.ordered_data_keys:
            # Skip internal keys
            if key.startswith('_'):
                continue

            # Check if all values for this key are numerical
            values = [r.get(key) for r in self.records[:100] if key in r]
            if values and all(isinstance(v, (int, float)) for v in values):
                numerical_keys.append(key)

        return numerical_keys  # No sorting - preserves catalog order
    
    def calculate_statistics(self, key, session_id=None):
        """Calculate statistics for a numerical key"""
        # Get filtered records
        records = self.get_records(session_id=session_id)
        
        # Extract values
        values = []
        for record in records:
            if key in record and isinstance(record[key], (int, float)):
                values.append(record[key])
        
        if not values:
            return None
        
        # Calculate statistics
        values_array = np.array(values)
        
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q1': float(np.percentile(values_array, 25)),
            'q3': float(np.percentile(values_array, 75))
        }
    
    def get_timestamp_range(self):
        """Get timestamp range of loaded data"""
        if not self.records:
            return None
        
        timestamps = [r.get('_timestamp_ms', 0) for r in self.records]
        
        return {
            'min': min(timestamps),
            'max': max(timestamps),
            'duration_ms': max(timestamps) - min(timestamps)
        }
    
    def get_timeline_data(self, key, session_id=None):
        """Get time series data for a specific key"""
        records = self.get_records(session_id=session_id)
        
        timeline_data = []
        for record in records:
            if key in record and '_timestamp_ms' in record:
                timeline_data.append({
                    'timestamp': record['_timestamp_ms'],
                    'value': record[key],
                    'index': record.get('_index', 0)
                })
        
        return timeline_data
    
    def get_deleted_indexes(self):
        """Get list of deleted indexes"""
        return self.deleted_indexes
    
    def get_image_paths_for_record(self, record):
        """Extract all image paths from a record"""
        image_paths = {}

        for key, value in record.items():
            if key.endswith('/image_array') and isinstance(value, str):
                image_paths[key] = value

        return image_paths

    def get_lidar_keys_for_record(self, record):
        """Extract all LiDAR distance array keys from a record"""
        lidar_keys = []
        for key in record.keys():
            if key.endswith('/distance_array'):
                lidar_keys.append(key)
        return lidar_keys

    def get_lidar_metadata(self):
        """Get LiDAR metadata from manifest.json"""
        if not self.data_path:
            return None

        manifest_path = os.path.join(self.data_path, 'manifest.json')
        if not os.path.exists(manifest_path):
            return None

        try:
            with open(manifest_path, 'r') as f:
                lines = f.readlines()

            metadata = {}
            # Try metadata line (line 3, index 2) first, then line index 3
            for line_idx in [2, 3]:
                if len(lines) > line_idx:
                    try:
                        data = json.loads(lines[line_idx])
                        if 'lidar_type' in data or 'lidar_data_points' in data:
                            metadata = data
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Fallback: scan all lines
            if not metadata:
                for line in lines:
                    try:
                        data = json.loads(line)
                        if 'lidar_type' in data or 'lidar_data_points' in data:
                            metadata = data
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

            if not metadata:
                return None

            return {
                'lidar_type': metadata.get('lidar_type', 'unknown'),
                'angle_start': metadata.get('lidar_angle_start', 0),
                'angle_end': metadata.get('lidar_angle_end', 360),
                'clockwise': metadata.get('lidar_clockwise', True),
                'data_points': metadata.get('lidar_data_points', 360),
            }
        except Exception:
            return None

    def get_lidar_array_size(self, npy_filename):
        """Get the number of points in a LiDAR distance array .npy file"""
        if not self.data_path:
            return None
        npy_path = os.path.join(self.data_path, 'lidar', npy_filename)
        if not os.path.exists(npy_path):
            return None
        data = np.load(npy_path)
        return int(data.shape[0])

    def load_lidar_data(self, npy_filename):
        """Load LiDAR distance data from a .npy file"""
        if not self.data_path:
            return None

        npy_path = os.path.join(self.data_path, 'lidar', npy_filename)
        if not os.path.exists(npy_path):
            return None

        data = np.load(npy_path)
        return data.tolist()

    def update_deleted_indexes(self, start_idx, end_idx):
        """Add range of indexes to deleted_indexes and update manifest file"""
        if not self.base_path:
            raise Exception("No data loaded")

        # Add indexes in range to deleted list
        new_deletes = set(self.deleted_indexes)
        for idx in range(start_idx, end_idx + 1):
            new_deletes.add(idx)

        self.deleted_indexes = sorted(list(new_deletes))

        # Update catalog_manifest
        if self.catalog_manifest is None:
            self.catalog_manifest = {}

        self.catalog_manifest['deleted_indexes'] = self.deleted_indexes

        # Write updated manifest back to file (use data_path to match load_data)
        manifest_path = os.path.join(self.data_path, 'manifest.json')

        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                lines = f.readlines()

            # Update line 4 (index 4) with new catalog_manifest
            if len(lines) > 4:
                lines[4] = json.dumps(self.catalog_manifest) + '\n'
            else:
                # Ensure we have enough lines
                while len(lines) < 4:
                    lines.append('\n')
                lines.append(json.dumps(self.catalog_manifest) + '\n')

            # Write back to file
            with open(manifest_path, 'w') as f:
                f.writelines(lines)

        # Update records in memory
        for record in self.records:
            if record.get('_absolute_index') in self.deleted_indexes:
                record['_is_deleted'] = True

        return self.deleted_indexes

    def clear_deleted_indexes(self, start_idx, end_idx):
        """Remove range of indexes from deleted_indexes and update manifest file"""
        if not self.base_path:
            raise Exception("No data loaded")

        # Remove indexes in range from deleted list
        indexes_to_remove = set(range(start_idx, end_idx + 1))
        new_deletes = [idx for idx in self.deleted_indexes if idx not in indexes_to_remove]

        self.deleted_indexes = sorted(new_deletes)

        # Update catalog_manifest
        if self.catalog_manifest is None:
            self.catalog_manifest = {}

        self.catalog_manifest['deleted_indexes'] = self.deleted_indexes

        # Write updated manifest back to file (use data_path to match load_data)
        manifest_path = os.path.join(self.data_path, 'manifest.json')

        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                lines = f.readlines()

            # Update line 4 (index 4) with new catalog_manifest
            if len(lines) > 4:
                lines[4] = json.dumps(self.catalog_manifest) + '\n'
            else:
                # Ensure we have enough lines
                while len(lines) < 4:
                    lines.append('\n')
                lines.append(json.dumps(self.catalog_manifest) + '\n')

            # Write back to file
            with open(manifest_path, 'w') as f:
                f.writelines(lines)

        # Update records in memory
        for record in self.records:
            if record.get('_absolute_index') in indexes_to_remove:
                record['_is_deleted'] = False

        return self.deleted_indexes