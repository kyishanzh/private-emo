import os
from pathlib import Path
import cv2
from PIL import Image
from typing import Callable
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

class PrivatizationRunner():

    def __init__(self, in_path : str, out_path : str, privatization : list, useCV : False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.privatization = privatization
        self.useCV = useCV

    def run(self, debug = False) -> None:
        for root, _, files in os.walk(self.in_path):
            root_path = Path(root)
            for file in files:
                # get input image
                file_path = root_path / file
                if file_path.suffix.lower() not in VALID_EXTENSIONS:
                    if debug:
                        print(f"Skipping non-image file: {file_path}")
                    continue
                # process
                out_img = np.array(Image.open(str(file_path)))
                if self.useCV:
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                for method in self.privatization:
                    out_img = method(out_img)
                # handle output file path stuff
                rel_path = file_path.relative_to(self.in_path)
                new_filename = f"{file_path.stem}_privatized{file_path.suffix}"
                output_subdir = self.out_path / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_file = output_subdir / new_filename

                # Save processed image
                if self.useCV:
                    cv2.imwrite(str(output_file), out_img)
                else:
                    out_img.save(output_file)

                os.remove(file_path)
                if debug:
                    print(f"Processed and saved: {output_file}")


# cwd = os.getcwd()
# print("Current working directory:", cwd)

# our input directories
celebA_dir = "CelebrityFacesDataset"
fer_dir = "data/real"
