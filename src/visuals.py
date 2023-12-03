import io 
from pathlib import Path
from typing import List 

import jax.numpy as np 
from matplotlib.patches import Rectangle 
import matplotlib.pyplot as plt 

from constants import ndarray 

def image_from_figure(figure, dpi: int=128) -> ndarray: 
    io_buf = io.BytesIO()
    figure.savefig(io_buf, format='raw', dpi=dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

def video_to_frames(video: ndarray, save_path: Path) -> None: 
    video: ndarray = np.transpose(video, (3, 0, 1, 2))
    num_frames: int = video.shape[0] 

    for i in range(num_frames): 
        plt.figure() 
        plt.imshow(video[i]) 
        plt.xticks([])
        plt.yticks([])
        plt.savefig(Path(save_path) / f"frame_{i}", bbox_inches='tight', pad_inches=0)
        plt.close()

def make_car(x: ndarray, u: ndarray, ax): 
    position: ndarray = x[:2]
    heading: float = np.rad2deg(x[-1])
    steering_angle: float = np.rad2deg(u[1])
    body_width: float = 0.2
    body_height: float = 0.12
    wheel_width: float = 0.05
    wheel_height: float = 0.005

    body = Rectangle(
        (position[0], position[1] - (body_height / 2)), 
        body_width, 
        body_height, 
        angle=heading, 
        rotation_point=tuple(position.tolist()), 
        edgecolor=(0, 0, 0, 0.8), 
        facecolor=(0, 0, 0, 0.15), 
        )

    f_center = np.array([x[0] + body_width * np.cos(x[-1]), x[1] + body_width * np.sin(x[-1])])
    s = body_height / 2 

    lwc = np.array([f_center[0] - (s * np.sin(x[-1])), f_center[1] + s * np.cos(x[-1])])
    rwc = np.array([f_center[0] + (s * np.sin(x[-1])), f_center[1] - s * np.cos(x[-1])])
    
    left_wheel = Rectangle(
        (lwc[0] - wheel_width/2, lwc[1] - wheel_height / 2), 
        wheel_width, 
        wheel_height, 
        angle=steering_angle + heading, 
        rotation_point='center',
        color='k'
    )

    right_wheel = Rectangle(
        (rwc[0] - wheel_width / 2, rwc[1] - wheel_height / 2), 
        wheel_width, 
        wheel_height, 
        angle=steering_angle + heading, 
        rotation_point='center',
        color='k'
    )
    ax.add_patch(body) 
    ax.add_patch(left_wheel) 
    ax.add_patch(right_wheel)

def render_scene(obstacles: List[ndarray], path: Path=None, obstacle_size: float=0.2, world_range=((-1., -1.), (2., 2.)), **kwargs):
  fig = plt.figure(figsize=(3, 3), dpi=kwargs.get("dpi", 128))
  ax = fig.add_subplot(111)
  plt.grid(False)

  for ob in obstacles:
    ax.add_patch(plt.Rectangle((ob[0], ob[1]), obstacle_size, obstacle_size, color='k', alpha=0.3))

  ax.set_xlim([world_range[0][0], world_range[1][0]])
  ax.set_ylim([world_range[0][1], world_range[1][1]])
  ax.set_xticks([world_range[0][0], world_range[1][0]])
  ax.set_yticks([world_range[0][1], world_range[1][1]])
  ax.set_aspect('equal')

  if path is None: 
      return fig, ax
  else: 
      plt.savefig(path) 
      plt.close()
