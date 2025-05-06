import os
from pathlib import Path
import cv2
from typing import Callable

class PrivatizationRunner():

    def __init__(self, in_path : str, out_path : str, privatization) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.privatization = privatization

    def run(self, debug = False) -> None:
        for root, _, files in os.walk(self.in_path):
            root_path = Path(root)
            for file in files:
                # get input image
                file_path = root_path / file
                # process
                out_img = self.privatization(str(file))
                # handle output file path stuff
                rel_path = file_path.relative_to(self.in_path)
                new_filename = f"{file_path.stem}_out{file_path.suffix}"
                output_subdir = self.out_path / rel_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_file = output_subdir / new_filename

                # Save processed image
                cv2.imwrite(str(output_file), out_img)
                if debug:
                    print(f"Processed and saved: {output_file}")


cwd = os.getcwd()
print("Current working directory:", cwd)

# our input directories
celebA_dir = "CelebrityFacesDataset"
fer_dir = "data/real"
