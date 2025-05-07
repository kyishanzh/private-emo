import cv2
import numpy as np
import mediapipe as mp
from PrivatizationRunner import PrivatizationRunner


out_dir = "mesh_test"
in_dir = "test_small_dataset"

def mesh(image):
# Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils


    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Blank canvas
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    # Generate unique colors for each landmark index
    def get_color_for_index(index):
        """Generate a unique color based on the index."""
        # Using a simple colormap to map index to color
        color_map = cv2.COLORMAP_JET
        return cv2.applyColorMap(np.array([[index]], dtype=np.uint8), color_map)[0][0].tolist()

    # Draw landmarks with unique colors
    if results.multi_face_landmarks:
        # Process the first face (if multiple faces exist)
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

    for i, landmark in enumerate(landmarks):
        x, y = int(landmark.x * w), int(landmark.y * h)

        # Generate a unique color for the landmark index
        color = get_color_for_index(i)

        # Directly set the color of the pixel corresponding to the landmark
        blank[y, x] = color  # Set the pixel at (x, y) to the generated color
    return blank


runner = PrivatizationRunner(in_dir, out_dir, [mesh], useCV = True)

runner.run()
