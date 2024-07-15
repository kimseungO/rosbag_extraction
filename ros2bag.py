import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class FisheyeToPlaneConverter(Node):

    def __init__(self, bag_file_path, output_directory):
        super().__init__('fisheye_to_plane_converter')

        self.bridge = CvBridge()
        self.bag_file_path = bag_file_path
        self.output_directory = output_directory

        # Precompute the coordinate mappings for fisheye correction
        self.image_info = self.initialize_image_info()

        # Create directories to save images
        self.create_directories()

        # Open the bag file
        self.reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)
        
        # Get all topics
        self.reader.has_next()
        self.topics = self.reader.get_all_topics_and_types()

    def initialize_image_info(self):
        image_info = {}
        
        params = [
            ("image0_in", 1920, 1920, 180, 90, 600, -40, 0),
            ("image0_under", 1920, 1920, 180, 90, 600, 0, 0),
            ("image0_out", 1920, 1920, 180, 90, 600, 40, 0),
            ("image2_in", 1920, 1920, 180, 90, 600, -40, 0),
            ("image2_under", 1920, 1920, 180, 90, 600, 0, 0),
            ("image2_out", 1920, 1920, 180, 90, 600, 40, 0),
            ("image4_in", 1920, 1920, 180, 90, 600, -40, 0),
            ("image4_under", 1920, 1920, 180, 90, 600, 0, 0),
            ("image4_out", 1920, 1920, 180, 90, 600, 40, 0),
            ("image6_in", 1920, 1920, 180, 90, 600, -40, 0),
            ("image6_under", 1920, 1920, 180, 90, 600, 0, 0),
            ("image6_out", 1920, 1920, -180, 90, 600, -25, 0),
            ("image8_in", 1920, 1920, 180, 90, 600, 40, 0),
            ("image8_under", 1920, 1920, 180, 90, 600, 0, 0),
            ("image8_out", 1920, 1920, 180, 90, 600, -28, 0)
        ]
        
        for name, ih, iw, i_fov, o_fov, o_sz, o_u, o_v in params:
            coor_x, coor_y = self.fisheye_to_plane_info(ih, iw, i_fov, o_fov, o_sz, o_u, o_v)
            image_info[name] = {"coor_x": coor_x, "coor_y": coor_y}
        
        return image_info

    def fisheye_to_plane_info(self, ih, iw, i_fov, o_fov, o_sz, o_u, o_v):
        i_fov = i_fov * np.pi / 180
        o_fov = o_fov * np.pi / 180
        o_u = o_u * np.pi / 180
        o_v = o_v * np.pi / 180

        x_grid, y_grid, z_grid = self.grid_in_3d_to_project(o_fov, o_sz, o_u, o_v)
        theta = np.arctan2(y_grid, x_grid)
        c_grid = np.sqrt(x_grid**2 + y_grid**2)
        rho = np.arctan2(c_grid, z_grid)
        r = rho * min(ih, iw) / i_fov
        coor_x = r * np.cos(theta) + iw / 2
        coor_y = r * np.sin(theta) + ih / 2

        return coor_x.astype(np.float32), coor_y.astype(np.float32)

    def grid_in_3d_to_project(self, o_fov, o_sz, o_u, o_v):
        z = 1
        L = np.tan(o_fov / 2) / z
        x = np.linspace(L, -L, num=o_sz, dtype=np.float64)
        y = np.linspace(-L, L, num=o_sz, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.ones_like(x_grid)

        Rx = self.get_rotation_matrix(o_v, [1, 0, 0])
        Ry = self.get_rotation_matrix(o_u, [0, 1, 0])

        xyz_grid = np.stack([x_grid, y_grid, z_grid], -1).dot(Rx).dot(Ry)

        return [xyz_grid[..., i] for i in range(3)]

    def get_rotation_matrix(self, rad, ax):
        ax = np.array(ax)
        ax = ax / np.sqrt((ax**2).sum())
        R = np.diag([np.cos(rad)] * 3)
        R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

        ax = ax * np.sin(rad)
        R = R + np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])

        return R

    def fisheye_remap(self, frame, image_info):
        transformed_image = cv2.remap(frame, image_info["coor_x"], image_info["coor_y"], interpolation=cv2.INTER_LINEAR)
        return np.fliplr(transformed_image)

    def process_image(self, frame, image_info_in, image_info_under, image_info_out):
        frame_new = self.prepare_frame(frame)
        
        cam_in = self.fisheye_remap(frame_new, image_info_in)
        cam_under = self.fisheye_remap(frame_new, image_info_under)
        cam_out = self.fisheye_remap(frame_new, image_info_out)

        cam_out = cv2.rotate(cam_out, cv2.ROTATE_90_CLOCKWISE)
        cam_in = cv2.rotate(cam_in, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return cam_in, cam_under, cam_out

    def prepare_frame(self, frame):
        w = frame.shape[1]
        h = frame.shape[0]
        black = np.zeros((int((w - h) / 2), w, 3), np.uint8)
        frame_new = cv2.vconcat([black, frame])
        frame_new = cv2.vconcat([frame_new, black])
        return frame_new

    def create_directories(self):
        directories = ['door1_in', 'door1_under', 'door1_out',
                       'door2_in', 'door2_under', 'door2_out',
                       'door3_in', 'door3_under', 'door3_out',
                       'bus1_in', 'bus1_under', 'bus1_out', 'bus2_in', 'bus2_under', 'bus2_out']
        for directory in directories:
            full_path = os.path.join(self.output_directory, directory)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

    def save_image(self, image, directory, frame_id):
        full_directory = os.path.join(self.output_directory, directory)
        filename = os.path.join(full_directory, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(filename, image)

    def process_bag_file(self):
        while self.reader.has_next():
            topic, data, t = self.reader.read_next()

            # Get the type of the message
            msg_type = self.get_message_type(topic)
            if msg_type is None:
                continue

            # Deserialize the message
            msg = self.deserialize_message(data, msg_type)

            if msg is None:
                continue

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if topic == '/camera0/image_raw':
                cam0_in, cam0_under, cam0_out = self.process_image(frame, 
                                                                   self.image_info["image0_in"], 
                                                                   self.image_info["image0_under"], 
                                                                   self.image_info["image0_out"])

                frame_id = msg.header.stamp.sec
                self.save_image(cam0_in, 'door1_in', frame_id)
                self.save_image(cam0_under, 'door1_under', frame_id)
                self.save_image(cam0_out, 'door1_out', frame_id)

            elif topic == '/camera4/image_raw':
                cam2_in, cam2_under, cam2_out = self.process_image(frame, 
                                                                   self.image_info["image2_in"], 
                                                                   self.image_info["image2_under"], 
                                                                   self.image_info["image2_out"])

                frame_id = msg.header.stamp.sec
                self.save_image(cam2_in, 'door2_in', frame_id)
                self.save_image(cam2_under, 'door2_under', frame_id)
                self.save_image(cam2_out, 'door2_out', frame_id)

            elif topic == '/camera6/image_raw':
                cam4_in, cam4_under, cam4_out = self.process_image(frame, 
                                                                   self.image_info["image4_in"], 
                                                                   self.image_info["image4_under"], 
                                                                   self.image_info["image4_out"])

                frame_id = msg.header.stamp.sec
                self.save_image(cam4_in, 'door3_in', frame_id)
                self.save_image(cam4_under, 'door3_under', frame_id)
                self.save_image(cam4_out, 'door3_out', frame_id)

            elif topic == '/camera2/image_raw':
                cam6_in, cam6_under, cam6_out = self.process_image(frame, 
                                                                   self.image_info["image6_in"], 
                                                                   self.image_info["image6_under"], 
                                                                   self.image_info["image6_out"])

                frame_id = msg.header.stamp.sec
                self.save_image(cam6_in, 'bus1_in', frame_id)
                self.save_image(cam6_under, 'bus1_under', frame_id)
                cam6_out = cv2.rotate(cam6_out, cv2.ROTATE_180)
                self.save_image(cam6_out, 'bus1_out', frame_id)

            elif topic == '/camera8/image_raw':
                cam8_in, cam8_under, cam8_out = self.process_image(frame, 
                                                                   self.image_info["image8_in"], 
                                                                   self.image_info["image8_under"], 
                                                                   self.image_info["image8_out"])

                cam8_in = cv2.rotate(cam8_in, cv2.ROTATE_180)
                cam8_out = cv2.rotate(cam8_out, cv2.ROTATE_180)

                frame_id = msg.header.stamp.sec
                self.save_image(cam8_in, 'bus2_in', frame_id)
                self.save_image(cam8_under, 'bus2_under', frame_id)
                self.save_image(cam8_out, 'bus2_out', frame_id)

    def get_message_type(self, topic):
        for topic_info in self.topics:
            if topic_info.name == topic:
                return topic_info.type
        return None

    def deserialize_message(self, data, msg_type):
        try:
            # Log the message type to understand its structure
            self.get_logger().info(f"Message/mnt/hdd/ type: {msg_type}")
            
            # Get message class using the message type
            msg_class = get_message(msg_type)
            msg = deserialize_message(data, msg_class)
            return msg
        except Exception as e:
            self.get_logger().error(f"Failed to deserialize message: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)

    bag_file_path = './scne1_best/rosbag2_2024_04_05-13_25_30_0.db3'
    output_directory = './output/images/set2_scne11'
    node = FisheyeToPlaneConverter(bag_file_path, output_directory)

    node.process_bag_file()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
