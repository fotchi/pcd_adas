#!/usr/bin/env python3
"""
Object Detection Node — YOLOv5 Integration

Autonomous driving perception node for real-time object detection using YOLOv5.
Integrates with CARLA simulator via ROS topics.

Subscriptions:
    - /carla/camera/rgb (sensor_msgs/Image): Camera feed from CARLA

Publications:
    - /adas/detection (std_msgs/String): CSV list of detected objects
    - /adas/detection_image (sensor_msgs/Image): Annotated detection visualization

Parameters:
    - ~model_path (str): Path to custom YOLOv5 .pt model. If not provided or invalid,
                        uses default YOLOv5s model
    - ~confidence_threshold (float): Detection confidence threshold [0.0-1.0]
    - ~inference_size (int): Input inference size in pixels
    - ~enabled_classes (list): List of COCO class indices to detect

Author: ADAS Development Team
License: MIT
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ============================================================================
# Constants & Configuration
# ============================================================================

@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    # Model parameters
    DEFAULT_MODEL: str = 'yolov5s'
    CONFIDENCE_THRESHOLD: float = 0.45
    INFERENCE_SIZE: int = 416
    DEVICE: str = 'auto'
    
    # Enabled COCO classes
    # 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic_light, 11=stop_sign
    ENABLED_CLASSES: List[int] = None
    
    # Color map for visualization (BGR format)
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = None
    
    # ROS topic names
    CAMERA_TOPIC: str = '/carla/camera/rgb'
    DETECTION_PUB_TOPIC: str = '/adas/detection'
    IMAGE_PUB_TOPIC: str = '/adas/detection_image'
    
    # Publisher queue sizes
    QUEUE_SIZE: int = 1
    
    # Logging
    LOG_THROTTLE: float = 1.0  # seconds
    
    def __post_init__(self):
        if self.ENABLED_CLASSES is None:
            self.ENABLED_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
        if self.CLASS_COLORS is None:
            self.CLASS_COLORS = {
                'person': (0, 0, 255),          # Red
                'bicycle': (0, 255, 0),         # Green
                'car': (255, 165, 0),           # Orange
                'motorcycle': (0, 165, 255),    # Orange-Red
                'bus': (255, 165, 0),           # Orange
                'truck': (255, 165, 0),         # Orange
                'traffic light': (0, 255, 255), # Yellow
                'stop sign': (0, 0, 200),       # Dark Red
                'unknown': (128, 128, 128),     # Gray
            }


# ============================================================================
# Object Detection Node
# ============================================================================

class ObjectDetectionNode:
    """
    YOLOv5-based object detection node for autonomous driving perception.
    
    Handles:
    - Model loading (custom or default)
    - Camera frame processing
    - Real-time object detection
    - Visualization with bounding boxes
    - Publication of results
    """
    
    # Throttle intervals for logging (seconds)
    LOG_THROTTLE_INTERVAL = 1.0
    ERROR_LOG_THROTTLE = 5.0
    
    def __init__(self):
        """Initialize the Object Detection Node"""
        # Initialize ROS node
        rospy.init_node('object_detection_node', anonymous=False)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.bridge = CvBridge()
        self.model = None
        self.model_loaded = False
        self.device = 'cpu'
        self._last_log_time = {}
        
        # Load model
        self._load_model()
        
        # Initialize publishers
        self.pub_detection = rospy.Publisher(
            self.config.DETECTION_PUB_TOPIC,
            String,
            queue_size=self.config.QUEUE_SIZE
        )
        self.pub_detection_image = rospy.Publisher(
            self.config.IMAGE_PUB_TOPIC,
            Image,
            queue_size=self.config.QUEUE_SIZE
        )
        
        # Initialize subscriber
        self.sub_camera = rospy.Subscriber(
            self.config.CAMERA_TOPIC,
            Image,
            self._camera_callback,
            queue_size=1,
            buff_size=52428800  # 50MB buffer for high-res images
        )
        
        rospy.loginfo(f"✅ Object Detection Node initialized successfully")
        rospy.loginfo(f"   Camera topic: {self.config.CAMERA_TOPIC}")
        rospy.loginfo(f"   Detection topic: {self.config.DETECTION_PUB_TOPIC}")
        rospy.loginfo(f"   Visualization topic: {self.config.IMAGE_PUB_TOPIC}")
    
    @staticmethod
    def _load_config() -> DetectionConfig:
        """
        Load configuration from ROS parameters and defaults.
        
        Returns:
            DetectionConfig: Configuration object
        """
        config = DetectionConfig()
        
        # Override defaults with ROS parameters
        config.CONFIDENCE_THRESHOLD = rospy.get_param(
            '~confidence_threshold',
            config.CONFIDENCE_THRESHOLD
        )
        config.INFERENCE_SIZE = rospy.get_param(
            '~inference_size',
            config.INFERENCE_SIZE
        )
        config.ENABLED_CLASSES = rospy.get_param(
            '~enabled_classes',
            config.ENABLED_CLASSES
        )
        config.DEVICE = rospy.get_param('~device', config.DEVICE)

        config.CONFIDENCE_THRESHOLD = max(0.0, min(1.0, float(config.CONFIDENCE_THRESHOLD)))
        config.INFERENCE_SIZE = max(160, int(config.INFERENCE_SIZE))
        
        rospy.loginfo(f"Configuration loaded:")
        rospy.loginfo(f"  Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
        rospy.loginfo(f"  Inference size: {config.INFERENCE_SIZE}x{config.INFERENCE_SIZE}")
        rospy.loginfo(f"  Enabled classes: {config.ENABLED_CLASSES}")
        
        return config

    def _load_model(self) -> bool:
        """
        Load YOLOv5 model from custom path or default.
        
        Tries to load a custom model specified via ROS parameter.
        Falls back to default YOLOv5s model if custom path is invalid.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            import torch # pyright: ignore[reportMissingImports]
            self._select_device(torch)
            
            model_path = rospy.get_param('~model_path', None)
            
            # Attempt to load custom model
            if model_path and os.path.exists(model_path):
                rospy.loginfo(f"Loading custom model from: {model_path}")
                try:
                    self.model = torch.hub.load(
                        'ultralytics/yolov5',
                        'custom',
                        path=model_path,
                        force_reload=False
                    )
                    self._configure_model()
                    rospy.loginfo(f"✅ Custom model loaded successfully")
                    self.model_loaded = True
                    return True
                except Exception as e:
                    rospy.logwarn(f"Failed to load custom model: {e}")
                    rospy.logwarn(f"Falling back to default YOLOv5s model")
            
            # Load default model
            rospy.loginfo(f"Loading default YOLOv5s model...")
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                self.config.DEFAULT_MODEL,
                pretrained=True,
                force_reload=False
            )

            self._configure_model()
            
            rospy.loginfo(f"✅ YOLOv5 model loaded successfully")
            self.model_loaded = True
            return True
            
        except ImportError as e:
            rospy.logerr(f"PyTorch not installed: {e}")
            rospy.logerr(f"Install with: pip3 install torch torchvision")
            return False
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            rospy.logerr(f"Model loading error details: {type(e).__name__}: {str(e)}")
            return False

    def _select_device(self, torch) -> None:
        device_param = str(self.config.DEVICE).lower()
        if device_param in ('auto', 'cuda') and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def _configure_model(self) -> None:
        if self.model is None:
            return
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'conf'):
            self.model.conf = self.config.CONFIDENCE_THRESHOLD
        if hasattr(self.model, 'classes'):
            self.model.classes = self.config.ENABLED_CLASSES
    
    def _camera_callback(self, msg: Image) -> None:
        """
        Process incoming camera frame and perform object detection.
        
        Args:
            msg (sensor_msgs/Image): Camera frame message
        """
        # Skip if model not loaded
        if not self.model_loaded or self.model is None:
            return
        
        try:
            # Convert ROS image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self._log_throttled('cv_bridge_error', f"CvBridge error: {e}", level='error')
            return
        except Exception as e:
            self._log_throttled('frame_error', f"Frame processing error: {e}", level='error')
            return
        
        # Perform detection and publish results
        try:
            detections_df = self._run_inference(frame)
            detections_list = self._extract_labels(detections_df)
            annotated_frame = self._annotate_frame(frame, detections_df)

            # Publish results
            self._publish_results(detections_list, annotated_frame)

            # Log detections
            if detections_list:
                det_str = ', '.join(detections_list)
                self._log_throttled('detection', f"Detected: {det_str}", level='info')

        except Exception as e:
            self._log_throttled('detection_error', f"Detection error: {e}", level='error')
    
    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLOv5 inference and return detections dataframe.

        Args:
            frame (np.ndarray): Input image (BGR format)

        Returns:
            pandas.DataFrame or None: Detection dataframe with bbox/labels
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame, size=self.config.INFERENCE_SIZE)

            if len(results) > 0 and hasattr(results, 'pandas'):
                return results.pandas().xyxy[0]
        except Exception as e:
            self._log_throttled('inference_error', f"Inference error: {e}", level='error')

        return None

    @staticmethod
    def _extract_labels(detections_df) -> List[str]:
        """
        Extract labels from detection dataframe.

        Args:
            detections_df: YOLOv5 detection dataframe

        Returns:
            List[str]: List of detected labels
        """
        if detections_df is None or len(detections_df) == 0:
            return []
        return detections_df['name'].tolist()
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections_df
    ) -> np.ndarray:
        """
        Add bounding boxes and labels to frame using model results.
        
        Args:
            frame (np.ndarray): Input image (BGR format)
            detections_df: YOLOv5 detection dataframe
        
        Returns:
            np.ndarray: Annotated image
        """
        annotated = frame.copy()
        
        try:
            if detections_df is not None and len(detections_df) > 0:
                for _, row in detections_df.iterrows():
                    x1, y1 = int(row['xmin']), int(row['ymin'])
                    x2, y2 = int(row['xmax']), int(row['ymax'])
                    label = str(row['name'])
                    confidence = float(row['confidence'])

                    color = self.config.CLASS_COLORS.get(
                        label,
                        self.config.CLASS_COLORS['unknown']
                    )

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    label_text = f"{label} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1

                    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                    text_x, text_y = x1, max(y1 - 5, text_size[1])

                    cv2.rectangle(
                        annotated,
                        (text_x, text_y - text_size[1] - 4),
                        (text_x + text_size[0] + 4, text_y + 2),
                        color,
                        -1
                    )

                    cv2.putText(
                        annotated,
                        label_text,
                        (text_x + 2, text_y - 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )
        
        except Exception as e:
            self._log_throttled('annotation_error', f"Annotation error: {e}", level='error')
        
        return annotated
    
    def _publish_results(
        self,
        detections: List[str],
        annotated_frame: np.ndarray
    ) -> None:
        """
        Publish detection results and visualization.
        
        Args:
            detections (List[str]): List of detected object labels
            annotated_frame (np.ndarray): Annotated image for visualization
        """
        # Publish detection list
        try:
            det_str = ','.join(detections) if detections else 'none'
            self.pub_detection.publish(String(data=det_str))
        except Exception as e:
            self._log_throttled('pub_detection_error', f"Detection publish error: {e}", level='error')
        
        # Publish annotated image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pub_detection_image.publish(img_msg)
        except CvBridgeError as e:
            self._log_throttled('pub_image_error', f"Image publish error: {e}", level='error')
        except Exception as e:
            self._log_throttled('pub_image_error', f"Image publish error: {e}", level='error')
    
    def _log_throttled(self, key: str, message: str, level: str = 'info') -> None:
        """
        Log message with throttling to prevent log spam.
        
        Args:
            key (str): Unique identifier for this log message
            message (str): Log message
            level (str): Log level ('debug', 'info', 'warn', 'error')
        """
        current_time = rospy.get_time()
        last_time = self._last_log_time.get(key, 0)
        
        throttle_interval = (
            self.ERROR_LOG_THROTTLE if level == 'error'
            else self.LOG_THROTTLE_INTERVAL
        )
        
        if current_time - last_time >= throttle_interval:
            self._last_log_time[key] = current_time
            
            if level == 'debug':
                rospy.logdebug(message)
            elif level == 'info':
                rospy.loginfo(message)
            elif level == 'warn':
                rospy.logwarn(message)
            elif level == 'error':
                rospy.logerr(message)
    
    def shutdown(self) -> None:
        """Clean up resources on shutdown"""
        rospy.loginfo("Shutting down Object Detection Node")
        # Release model memory
        self.model = None
        rospy.loginfo("✅ Object Detection Node shut down successfully")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    try:
        node = ObjectDetectionNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
