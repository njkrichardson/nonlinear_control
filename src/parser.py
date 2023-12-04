import argparse 
from glob import glob 
from pathlib import Path 
from typing import List
import sys 

import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import tqdm 

from constants import ndarray 
from utils import LOG_DIRECTORY, setup_logger, serialize, DATA_DIRECTORY, deserialize, human_bytes_str
from visuals import render_scene, make_car, image_from_figure

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, required=True)
parser.add_argument("--gifs", action="store_true")
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--analyze-only", action="store_true") 
parser.add_argument("--num-video-limit", type=int, default=1_000)
parser.add_argument("--mpl-backend", default="Agg", type=str) 

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def process_result(args, path: Path, i=0) -> ndarray: 
    results: dict = deserialize(path)  
    num_timesteps: int = results['T']
    obstacles: List[ndarray] = results['obstacles']
    X: ndarray = results['X']
    U: ndarray = results['U']
    x0: ndarray = results['x0'] 
    goal_default: ndarray = results['goal_default'] 

    def render_state(X: ndarray, U: ndarray, t: int) -> ndarray: 
        fig, ax = render_scene(obstacles, production=True)
        x = X[t, :2]
        dx = np.array([0.2 * np.sin(X[t, 2]), 0.2 * np.cos(X[t, 2])])
        y = x + dx 
        L = np.linalg.norm(y - x) 
        heading = np.arccos(dx[0] / L)
        make_car(np.array([x[0], x[1], heading]), np.array([U[t, 1], -U[t, 0]]), ax)
        ax.set_aspect('equal')

        image: ndarray = image_from_figure(fig)
        plt.close()

        return image 

    images: List[ndarray] = [] 

    for t in range(0, num_timesteps, 3): 
        image: ndarray = np.array(render_state(X, U, t)[..., :3])
        color_image: Image = Image.fromarray(image) 
        grayscale_image: ndarray = np.array(color_image.convert('L'))[..., None]
        images.append(image) 

    video: ndarray = np.array(images) 

    if args.gifs: 
        images = [Image.fromarray(im) for im in video]
        images[0].save(Path(args.save_path) / f"example_{i}.gif", save_all=True, append_images=images[1:], duration=50, loop=0)

    # channels, num_frames, height, width 
    return np.transpose(video, (3, 0, 1, 2))

def main(args): 
    # logging 
    log = setup_logger(__name__) 
    log.info("starting results parser") 

    # get candidate directories 
    results_directories: List[Path] = glob((Path(args.results_dir) / "*").as_posix()) 
    results: List[Path] = [] 

    for i, dir in enumerate(results_directories): 
        results_paths: List[Path] = glob((Path(dir) / "*.pkl").as_posix())
        results.extend(results_paths)


    if len(results) > args.num_video_limit: 
        results = results[:args.num_video_limit] # TODO, don't even collect them! 

    num_videos: int = len(results) 


    # process the first result to get the shapes correct and estimate the data size 
    if not args.gifs: 
        video: ndarray = process_result(args, results[0])

        frames_per_video: int = video.shape[1] 
        frame_height: int = video.shape[2] 
        frame_width: int = video.shape[3] 
        num_channels: int = video.shape[0] 
        video_num_bytes: int = video.size * video.itemsize 
        total_num_bytes: int = video_num_bytes * num_videos

        log.info("=" * 20 + "\tVIDEO STATS\t" + "="*20)
        log.info(f"{frames_per_video=}")
        log.info(f"{frame_height=}")
        log.info(f"{frame_width=}")
        log.info(f"{num_channels=}")
        log.info("=" * 20 + "\tVIDEO STATS\t" + "="*20)
        log.info("\n\n")

        log.info("=" * 20 + "\tESTIMATED MEMORY REQUIREMENTS\t" + "="*20)
        log.info(f"{num_videos=}")
        log.info(f"memory per video={human_bytes_str(video_num_bytes)}")
        log.info(f"total memory required={human_bytes_str(total_num_bytes)}")
        log.info("=" * 20 + "\tESTIMATED MEMORY REQUIREMENTS\t" + "="*20)
        log.info("\n\n")

    if args.analyze_only: 
        sys.exit(0)

    if not args.gifs: 
        videos: ndarray = np.empty((num_videos, num_channels, frames_per_video, frame_height, frame_width), dtype=video.dtype)
        videos[0] = video 
    i: int = 1

    for result in (progress_bar := tqdm.tqdm(results[1:])): 
        progress_bar.set_description(f"Processing {result=}")
        if not args.gifs: 
            video: ndarray = process_result(args, result, i=i) 
            videos[i] = video
        else: 
            process_result(args, result, i=i) 
        i += 1

    if not args.gifs: 
        save_path: Path = Path(args.save_path) / "videos.pkl"
        serialize(videos, save_path)
        log.info(f"Saved videos to {save_path.as_posix()}")

if __name__=="__main__": 
    args = parser.parse_args()
    matplotlib.use(f"{args.mpl_backend}")
    main(args) 
