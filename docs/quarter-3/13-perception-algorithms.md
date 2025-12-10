---
title: "13. Perception Algorithms"
sidebar_label: "13. Perception Algorithms"
sidebar_position: 5
---

# Chapter 13: Perception Algorithms

## Advanced Machine Learning and AI for Robot Perception

Perception algorithms represent the intelligence layer that transforms raw sensor data into meaningful understanding of the environment. This chapter explores advanced machine learning techniques, deep learning approaches, and cognitive algorithms that enable humanoid robots to perceive, reason about, and interact with their world in increasingly sophisticated ways.

## 🎯 Chapter Overview

### Learning Objectives
By the end of this chapter, you will be able to:
- Implement traditional feature extraction and matching algorithms
- Apply deep learning for object detection and recognition
- Develop semantic segmentation and scene understanding systems
- Create attention mechanisms for focused perception
- Build reinforcement learning agents for adaptive perception
- Design context-aware perception systems

### Prerequisites
- **Chapter 11**: Computer Vision fundamentals
- **Chapter 12**: Sensor Fusion techniques
- Basic machine learning concepts (supervised/unsupervised learning)
- Python programming with NumPy, Scikit-learn
- Understanding of neural networks basics

### Required Software and Libraries
- PyTorch or TensorFlow for deep learning
- OpenCV for computer vision operations
- Scikit-learn for traditional ML algorithms
- Matplotlib and Plotly for visualization
- CUDA-capable GPU for deep learning training

---

## 🧠 13.1 Feature Extraction and Description

### Traditional Feature Descriptors

#### **Scale-Invariant Feature Transform (SIFT)**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SIFTFeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def extract_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=0.75):
        """Match SIFT features using Lowe's ratio test"""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def visualize_matches(self, img1, kp1, img2, kp2, matches):
        """Visualize feature matches between two images"""
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return match_img

# Usage example
extractor = SIFTFeatureExtractor()

# Load two images
img1 = cv2.imread('scene1.jpg')
img2 = cv2.imread('scene2.jpg')

# Extract features
kp1, desc1 = extractor.extract_features(img1)
kp2, desc2 = extractor.extract_features(img2)

# Match features
matches = extractor.match_features(desc1, desc2)

# Visualize
result = extractor.visualize_matches(img1, kp1, img2, kp2, matches)
cv2.imshow('SIFT Matches', result)
cv2.waitKey(0)
```

#### **Oriented FAST and Rotated BRIEF (ORB)**
```python
class ORBFeatureExtractor:
    def __init__(self, n_features=500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, image):
        """Extract ORB features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match ORB features"""
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def ransac_fundamental_matrix(self, kp1, kp2, matches):
        """Estimate fundamental matrix using RANSAC"""
        # Extract matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # Select inlier matches
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return F, inlier_matches

    def visualize_epipolar_lines(self, img1, img2, kp1, kp2, F, matches):
        """Visualize epipolar lines for stereo correspondence"""
        # Select points for visualization
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:20]])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:20]])

        # Find epilines corresponding to points in second image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)

        # Draw epilines on first image
        img5, img6 = self.draw_epilines(img1, img2, lines1, pts1, pts2)

        return img5, img6

    def draw_epilines(self, img1, img2, lines, pts1, pts2):
        """Helper function to draw epipolar lines"""
        r, c = img1.shape[:2]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
            img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
        return img1, img2
```

### Advanced Feature Learning

#### **Learned Feature Descriptors with Deep Learning**
```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class LearnedFeatureDescriptor(nn.Module):
    def __init__(self, input_size=32, descriptor_size=128):
        super(LearnedFeatureDescriptor, self).__init__()

        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Descriptor head
        self.descriptor_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, descriptor_size),
            nn.Tanh()  # Normalize to [-1, 1]
        )

        # Key point detection head
        self.keypoint_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 65),  # 8x8 grid + 1 for background
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_net(x)
        features_flat = features.view(features.size(0), -1)

        # Generate descriptor
        descriptor = self.descriptor_head(features_flat)
        descriptor = F.normalize(descriptor, p=2, dim=1)

        # Detect keypoint
        keypoint_heatmap = self.keypoint_head(features_flat)
        keypoint_heatmap = keypoint_heatmap.view(-1, 8, 8)

        return descriptor, keypoint_heatmap

def extract_patches(image, patch_size=32, stride=16):
    """Extract patches from image for feature learning"""
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

    return np.array(patches)

def train_feature_descriptor(model, train_images, epochs=50):
    """Train learned feature descriptor"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0

        for img_pair in train_images:
            img1, img2 = img_pair

            # Extract patches
            patches1 = extract_patches(img1)
            patches2 = extract_patches(img2)

            # Convert to tensors
            patches1 = torch.from_numpy(patches1).float().unsqueeze(1)
            patches2 = torch.from_numpy(patches2).float().unsqueeze(1)

            # Forward pass
            desc1, _ = model(patches1)
            desc2, _ = model(patches2)

            # Compute contrastive loss
            loss = contrastive_loss(desc1, desc2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_images):.4f}")

def contrastive_loss(desc1, desc2, margin=1.0):
    """Contrastive loss for feature descriptor learning"""
    # Compute similarity matrix
    similarity = torch.mm(desc1, desc2.t())

    # Positive pairs (diagonal)
    positive = torch.diag(similarity)

    # Negative pairs (off-diagonal)
    negative = similarity - 2 * torch.eye(similarity.size(0))

    # Compute loss
    loss = torch.mean(torch.relu(margin - positive)) + torch.mean(negative)

    return loss
```

---

## 🤖 13.2 Deep Learning for Object Detection

### Convolutional Neural Networks for Detection

#### **YOLO (You Only Look Once) Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLODetection(nn.Module):
    def __init__(self, num_classes=80, input_size=416, grid_size=13):
        super(YOLODetection, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.grid_size = grid_size
        self.num_boxes = 5  # Number of bounding boxes per grid cell

        # Backbone network (simplified DarkNet)
        self.backbone = self._create_backbone()

        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, (num_classes + 5) * self.num_boxes, kernel_size=1)
        )

    def _create_backbone(self):
        """Create simplified DarkNet backbone"""
        layers = []

        # Convolutional layers
        in_channels = 3
        filters = [32, 64, 128, 256, 512, 1024]

        for out_channels in filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels // 2, 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            ])
            in_channels = out_channels

            # Add pooling for some layers
            if out_channels < 512:
                layers.append(nn.MaxPool2d(2))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Detection head
        detections = self.detection_head(features)

        # Reshape to grid format
        batch_size = detections.size(0)
        detections = detections.view(
            batch_size,
            self.num_boxes,
            self.num_classes + 5,
            self.grid_size,
            self.grid_size
        )

        # Apply sigmoid to confidences and classes
        detections[..., 4] = torch.sigmoid(detections[..., 4])  # Object confidence
        detections[..., 5:] = torch.sigmoid(detections[..., 5:])  # Class probabilities

        return detections

    def decode_predictions(self, detections, confidence_threshold=0.5, nms_threshold=0.4):
        """Decode YOLO predictions"""
        batch_size = detections.size(0)
        results = []

        for i in range(batch_size):
            pred = detections[i]

            # Filter by confidence
            mask = pred[..., 4] > confidence_threshold
            pred = pred[mask]

            if pred.size(0) == 0:
                results.append([])
                continue

            # Convert box coordinates
            boxes = self.convert_boxes(pred[..., :4])

            # Combine confidence and class scores
            scores = pred[..., 4:5] * pred[..., 5:]
            class_scores, class_ids = torch.max(scores, dim=1)

            # Combine class scores with box scores
            final_scores = pred[..., 4] * class_scores

            # Apply NMS
            keep = self.non_max_suppression(boxes, final_scores, nms_threshold)

            results.append({
                'boxes': boxes[keep],
                'scores': final_scores[keep],
                'classes': class_ids[keep]
            })

        return results

    def convert_boxes(self, boxes):
        """Convert YOLO box format to (x1, y1, x2, y2)"""
        # Input boxes are in (center_x, center_y, width, height) format
        center_x = boxes[..., 0]
        center_y = boxes[..., 1]
        width = boxes[..., 2]
        height = boxes[..., 3]

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def non_max_suppression(self, boxes, scores, threshold=0.4):
        """Non-maximum suppression"""
        if boxes.size(0) == 0:
            return torch.empty(0, dtype=torch.long)

        # Compute box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort by scores
        _, indices = scores.sort(descending=True)

        keep = []
        while indices.size(0) > 0:
            # Select the box with highest score
            current = indices[0]
            keep.append(current)

            if indices.size(0) == 1:
                break

            # Compute IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]

            # Calculate IoU
            xx1 = torch.max(current_box[0], remaining_boxes[:, 0])
            yy1 = torch.max(current_box[1], remaining_boxes[:, 1])
            xx2 = torch.min(current_box[2], remaining_boxes[:, 2])
            yy2 = torch.min(current_box[3], remaining_boxes[:, 3])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)

            intersection = w * h
            union = areas[current] + areas[indices[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU below threshold
            indices = indices[1:][iou < threshold]

        return torch.tensor(keep)
```

### Transfer Learning for Custom Object Detection

#### **Faster R-CNN with Pre-trained Backbone**
```python
import torchvision.models as models
import torchvision.transforms as transforms

class CustomObjectDetector:
    def __init__(self, num_classes, pretrained_backbone=True):
        self.num_classes = num_classes

        # Load pre-trained Faster R-CNN
        if pretrained_backbone:
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

        # Modify the classifier for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train_step(self, images, targets):
        """Single training step"""
        # Preprocess images
        images = [self.transform(img) for img in images]

        # Forward pass
        self.model.train()
        loss_dict = self.model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        return losses, loss_dict

    def predict(self, image, confidence_threshold=0.5):
        """Make predictions on a single image"""
        self.model.eval()

        with torch.no_grad():
            # Preprocess image
            img_tensor = self.transform(image)

            # Make prediction
            prediction = self.model([img_tensor])

            # Filter by confidence
            keep = prediction[0]['scores'] > confidence_threshold

            filtered_prediction = {
                'boxes': prediction[0]['boxes'][keep],
                'labels': prediction[0]['labels'][keep],
                'scores': prediction[0]['scores'][keep]
            }

            return filtered_prediction

def create_custom_dataset(image_dir, annotation_dir):
    """Create custom dataset for training"""
    # This would typically use a standard dataset format like COCO
    # or a custom format with bounding box annotations

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, annotation_dir, transforms=None):
            self.image_dir = image_dir
            self.annotation_dir = annotation_dir
            self.transforms = transforms

            # Load image paths and annotations
            self.images = self._load_images()
            self.annotations = self._load_annotations()

        def _load_images(self):
            """Load image file paths"""
            import glob
            return sorted(glob.glob(f"{self.image_dir}/*.jpg"))

        def _load_annotations(self):
            """Load annotations"""
            # Implement annotation loading logic
            # Return list of dictionaries with bounding boxes and labels
            return []

        def __getitem__(self, idx):
            image = cv2.imread(self.images[idx])
            annotation = self.annotations[idx]

            if self.transforms:
                image = self.transforms(image)

            target = {
                'boxes': torch.tensor(annotation['boxes'], dtype=torch.float32),
                'labels': torch.tensor(annotation['labels'], dtype=torch.int64)
            }

            return image, target

        def __len__(self):
            return len(self.images)

    return CustomDataset(image_dir, annotation_dir)
```

---

## 🎨 13.3 Semantic Segmentation

### Fully Convolutional Networks

#### **U-Net Architecture for Semantic Segmentation**
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Center
        self.center = self._block(512, 1024)

        # Decoder (upsampling)
        self.dec4 = self._block(1024 + 512, 512)
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, out_channels):
        """Create a basic block with two convolutions"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Center
        center = self.center(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        # Final classification
        output = self.final(dec1)

        return output

def train_unet(model, train_loader, val_loader, epochs=100, device='cuda'):
    """Train U-Net for semantic segmentation"""
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        iou_score = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate IoU
                iou_score += calculate_iou(outputs, masks)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"IoU Score: {iou_score/len(val_loader):.4f}")
        print("-" * 50)

def calculate_iou(outputs, masks):
    """Calculate Intersection over Union for segmentation"""
    _, predicted = torch.max(outputs, dim=1)

    intersection = (predicted & masks).float().sum((1, 2))
    union = (predicted | masks).float().sum((1, 2))

    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero

    return iou.mean().item()
```

### Real-time Semantic Segmentation

#### **BiSeNet for Efficient Segmentation**
```python
class BiSeNet(nn.Module):
    def __init__(self, num_classes=19):
        super(BiSeNet, self).__init__()

        # Spatial path (high-resolution features)
        self.spatial_path = SpatialPath()

        # Context path (semantic features)
        self.context_path = ContextPath()

        # Feature fusion module
        self.ffm = FeatureFusionModule(256, 256)

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Extract spatial features
        spatial_features = self.spatial_path(x)

        # Extract context features
        context_low, context_high = self.context_path(x)

        # Upsample context features
        context_high_up = F.interpolate(context_high, size=spatial_features.shape[2:],
                                       mode='bilinear', align_corners=False)

        # Fuse features
        fused_features = self.ffm(torch.cat([spatial_features, context_high_up], dim=1))

        # Final segmentation
        output = self.seg_head(fused_features)

        return output

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()

        # Lightweight backbone (e.g., MobileNetV2)
        self.backbone = self._create_lightweight_backbone()

        # Global context
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(512, 256, kernel_size=1)

        # Attention module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

    def _create_lightweight_backbone(self):
        """Create a lightweight backbone for context extraction"""
        # Simplified MobileNet-like structure
        layers = []

        in_channels = 3
        channels = [16, 24, 32, 64, 96, 160, 320]

        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            ])
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        # Extract features at multiple scales
        features = self.backbone(x)

        # Global context
        global_context = self.global_avg_pool(features)
        global_context = self.global_conv(global_context)

        # Attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Upsample for low-resolution path
        low_res = F.interpolate(attended_features, scale_factor=8,
                              mode='bilinear', align_corners=False)

        return low_res, global_context

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_extend = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Reduce channels
        reduced = self.conv_reduce(x)

        # Extend channels
        extended = self.conv_extend(x)

        # Combine
        combined = reduced + extended

        # Normalize and activate
        output = self.activation(self.batch_norm(combined))

        return output
```

---

## 🔍 13.4 Attention Mechanisms

### Visual Attention for Focused Perception

#### **Self-Attention for Computer Vision**
```python
class SelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(SelfAttention, self).__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels

        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate Q, K, V
        q = self.query(x).view(batch_size, -1, height * width)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # Attention weights
        attention = torch.bmm(q.transpose(1, 2), k)  # (B, H*W, H*W)
        attention = self.softmax(attention / (self.key_channels ** 0.5))

        # Apply attention to values
        out = torch.bmm(v, attention)
        out = out.view(batch_size, channels, height, width)

        # Residual connection
        out = self.out(out) + x

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average and max pooling
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)

        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Channel attention first
        x = self.channel_attention(x)

        # Then spatial attention
        x = self.spatial_attention(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(combined))

        return x * attention
```

### Multi-Head Attention for Vision Transformers

#### **Vision Transformer (ViT) Block**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        output = self.out_linear(context)

        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 d_model=768, n_heads=12, n_layers=12, n_classes=1000, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, d_model,
                                        kernel_size=patch_size, stride=patch_size)

        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model)
        )

        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Linear(d_model, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, d_model, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)

        # Add class token and position embedding
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        # Transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)

        # Classification
        cls_token = x[:, 0]
        logits = self.classifier(cls_token)

        return logits, attention_weights
```

---

## 🧪 13.5 Practical Project: Intelligent Scene Understanding

### Comprehensive Perception System

```python
class IntelligentSceneUnderstanding:
    def __init__(self):
        # Perception modules
        self.object_detector = self.load_object_detector()
        self.semantic_segmentation = self.load_segmentation_model()
        self.depth_estimation = self.load_depth_model()

        # Scene understanding components
        self.scene_graph_generator = SceneGraphGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.attention_system = AttentionSystem()

        # Knowledge base
        self.knowledge_base = self.load_knowledge_base()

    def understand_scene(self, image, depth_map=None):
        """Comprehensive scene understanding"""
        scene_analysis = {}

        # Object detection
        objects = self.object_detector.predict(image)
        scene_analysis['objects'] = objects

        # Semantic segmentation
        segmentation = self.semantic_segmentation.predict(image)
        scene_analysis['segmentation'] = segmentation

        # Depth estimation
        if depth_map is None:
            depth_map = self.depth_estimation.predict(image)
        scene_analysis['depth'] = depth_map

        # Build scene graph
        scene_graph = self.scene_graph_generator.build_graph(
            objects, segmentation, depth_map
        )
        scene_analysis['scene_graph'] = scene_graph

        # Contextual analysis
        context = self.context_analyzer.analyze(scene_graph, self.knowledge_base)
        scene_analysis['context'] = context

        # Attention-guided analysis
        attention_map = self.attention_system.compute_attention(image, objects)
        scene_analysis['attention'] = attention_map

        # Generate high-level description
        description = self.generate_scene_description(scene_analysis)
        scene_analysis['description'] = description

        return scene_analysis

    def generate_scene_description(self, scene_analysis):
        """Generate natural language description of the scene"""
        objects = scene_analysis['objects']
        context = scene_analysis['context']

        # Extract key information
        main_objects = [obj for obj in objects if obj['score'] > 0.5]
        scene_type = context.get('scene_type', 'unknown')
        location = context.get('location', 'unknown')
        activities = context.get('activities', [])

        # Generate description
        description = f"This is a {scene_type} scene "

        if location != 'unknown':
            description += f"in a {location}. "
        else:
            description += "with "

        # Describe main objects
        if main_objects:
            object_names = [obj['class'] for obj in main_objects[:5]]
            description += f"containing {', '.join(object_names)}. "

        # Describe activities
        if activities:
            description += f"Activities detected: {', '.join(activities)}. "

        # Add context about spatial relationships
        if 'spatial_relationships' in context:
            spatial_info = context['spatial_relationships']
            description += f"Spatial arrangement: {spatial_info}. "

        return description

class SceneGraphGenerator:
    def __init__(self):
        self.relation_types = ['above', 'below', 'next_to', 'inside', 'on_top_of', 'in_front_of']

    def build_graph(self, objects, segmentation, depth_map):
        """Build a scene graph from perception results"""
        graph = {
            'nodes': [],
            'edges': []
        }

        # Add object nodes
        obj_id = 0
        for obj in objects:
            node = {
                'id': obj_id,
                'type': obj['class'],
                'confidence': obj['score'],
                'bbox': obj['bbox'],
                'properties': self.extract_object_properties(obj, segmentation, depth_map)
            }
            graph['nodes'].append(node)
            obj_id += 1

        # Add spatial relationships
        edges = self.extract_spatial_relationships(graph['nodes'], depth_map)
        graph['edges'] = edges

        return graph

    def extract_object_properties(self, obj, segmentation, depth_map):
        """Extract detailed properties for an object"""
        bbox = obj['bbox']
        x1, y1, x2, y2 = bbox

        # Extract color information
        # (This would require access to the original image)
        color_info = self.extract_color_info(obj)

        # Extract depth information
        depth_info = self.extract_depth_info(obj, depth_map)

        # Extract size information
        size_info = self.extract_size_info(obj, depth_info)

        properties = {
            'color': color_info,
            'depth': depth_info,
            'size': size_info,
            'position': self.get_3d_position(obj, depth_info)
        }

        return properties

    def extract_spatial_relationships(self, nodes, depth_map):
        """Extract spatial relationships between objects"""
        relationships = []

        for i, obj1 in enumerate(nodes):
            for j, obj2 in enumerate(nodes):
                if i >= j:
                    continue

                # Calculate spatial relationship
                relation = self.calculate_spatial_relation(obj1, obj2, depth_map)
                if relation:
                    relationships.append(relation)

        return relationships

    def calculate_spatial_relation(self, obj1, obj2, depth_map):
        """Calculate spatial relationship between two objects"""
        pos1 = obj1['properties']['position']
        pos2 = obj2['properties']['position']

        if pos1 is None or pos2 is None:
            return None

        # Calculate relative position
        rel_pos = pos2 - pos1

        # Determine relationship type
        if abs(rel_pos[0]) < 0.5 and abs(rel_pos[1]) < 0.5:
            return {
                'type': 'same_location',
                'source': obj1['id'],
                'target': obj2['id'],
                'distance': np.linalg.norm(rel_pos)
            }
        elif rel_pos[1] < -0.5:
            return {
                'type': 'above',
                'source': obj1['id'],
                'target': obj2['id'],
                'distance': abs(rel_pos[1])
            }
        elif rel_pos[1] > 0.5:
            return {
                'type': 'below',
                'source': obj1['id'],
                'target': obj2['id'],
                'distance': abs(rel_pos[1])
            }
        elif rel_pos[0] < -0.5:
            return {
                'type': 'left_of',
                'source': obj1['id'],
                'target': obj2['id'],
                'distance': abs(rel_pos[0])
            }
        elif rel_pos[0] > 0.5:
            return {
                'type': 'right_of',
                'source': obj1['id'],
                'target': obj2['id'],
                'distance': abs(rel_pos[0])
            }

        return None

class ContextAnalyzer:
    def __init__(self):
        self.scene_patterns = self.load_scene_patterns()
        self.object_co_occurrence = self.load_object_co_occurrence()

    def analyze(self, scene_graph, knowledge_base):
        """Analyze scene context and make inferences"""
        analysis = {
            'scene_type': self.classify_scene_type(scene_graph),
            'location': self.infer_location(scene_graph),
            'activities': self.infer_activities(scene_graph),
            'anomalies': self.detect_anomalies(scene_graph),
            'intentions': self.infer_intentions(scene_graph)
        }

        return analysis

    def classify_scene_type(self, scene_graph):
        """Classify the type of scene"""
        objects = [node['type'] for node in scene_graph['nodes']]

        # Simple rule-based classification
        if 'chair' in objects and 'table' in objects and 'plate' in objects:
            return 'dining'
        elif 'bed' in objects or 'pillow' in objects:
            return 'bedroom'
        elif 'stove' in objects or 'refrigerator' in objects:
            return 'kitchen'
        elif 'person' in objects and 'laptop' in objects:
            return 'office'
        elif 'car' in objects or 'road' in objects:
            return 'outdoor'
        else:
            return 'general'

class AttentionSystem:
    def __init__(self):
        self.attention_weights = {}
        self.saliency_model = self.load_saliency_model()

    def compute_attention(self, image, objects):
        """Compute attention map for the scene"""
        # Bottom-up saliency
        saliency_map = self.saliency_model.compute_saliency(image)

        # Top-down attention based on objects
        object_attention = self.compute_object_attention(image, objects)

        # Combine attention maps
        combined_attention = self.combine_attention_maps(saliency_map, object_attention)

        return combined_attention

    def compute_object_attention(self, image, objects):
        """Compute top-down attention based on detected objects"""
        height, width = image.shape[:2]
        attention_map = np.zeros((height, width))

        for obj in objects:
            bbox = obj['bbox']
            confidence = obj['score']

            # Create Gaussian attention for each object
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            object_size = max(x2 - x1, y2 - y1)

            # Create Gaussian blob
            y, x = np.ogrid[:height, :width]
            gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (object_size / 4)**2))

            attention_map += confidence * gaussian

        # Normalize
        attention_map = attention_map / (np.max(attention_map) + 1e-8)

        return attention_map
```

---

## ✅ 13.6 Chapter Summary and Key Takeaways

### Core Concepts Covered
1. **Traditional Feature Extraction**: SIFT, ORB, and feature matching algorithms
2. **Deep Learning for Detection**: YOLO, Faster R-CNN, and transfer learning
3. **Semantic Segmentation**: U-Net, BiSeNet, and efficient architectures
4. **Attention Mechanisms**: Self-attention, multi-head attention, and CBAM
5. **Scene Understanding**: Scene graphs, context analysis, and attention systems
6. **Practical Applications**: Comprehensive intelligent scene understanding system

### Key Skills Developed
- Implementing traditional and modern feature extraction algorithms
- Training custom object detection models
- Building efficient semantic segmentation systems
- Applying attention mechanisms for focused perception
- Creating scene understanding and interpretation systems

### Common Challenges and Solutions
- **Real-time Performance**: Model optimization and efficient architectures
- **Dataset Requirements**: Synthetic data generation and data augmentation
- **Generalization**: Transfer learning and domain adaptation techniques
- **Computational Resources**: Efficient model design and deployment strategies

### Best Practices
- **Modular Design**: Separate perception components for maintainability
- **Performance Optimization**: Use efficient architectures and hardware acceleration
- **Robust Evaluation**: Comprehensive testing across diverse scenarios
- **Continuous Learning**: Model updates and adaptation to new environments

---

## 🚀 Next Steps

In the next chapter, we'll explore **Isaac Sim** (Chapter 14), where we'll dive into NVIDIA's advanced robotics simulation platform for creating realistic AI-powered simulation environments.

### Preparation for Next Chapter
- Install NVIDIA Isaac Sim and set up development environment
- Study 3D computer graphics and simulation concepts
- Review machine learning for simulation and robotics
- Explore synthetic data generation techniques

**Remember**: Perception algorithms are the bridge between raw sensor data and intelligent robot behavior. The techniques covered in this chapter form the foundation for creating truly intelligent and context-aware humanoid robots that can understand and interact with their environment in sophisticated ways! 🤖🧠🔍