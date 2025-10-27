import numpy as np

class TrafficImageDataset:
    """Dataset structure for multi-modal data"""
    def __init__(self, traffic_root, image_root):
        self.traffic_root = traffic_root
        self.image_root = image_root
        self.samples = []  # Store data path pairs (specific loading logic simplified)
        self.speed_max = 1.0  # Normalization parameter (calculation logic simplified)
        self.occ_max = 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Core data structure illustration (implementation simplified)
        traffic_data = np.load(self.samples[idx][0])  # [120, 3]
        image_data = np.load(self.samples[idx][1])    # [120, 128]
        target = np.load(self.samples[idx][2])        # [30, 2]
        # Preprocessing like normalization (details simplified)
        return traffic_data, image_data, target