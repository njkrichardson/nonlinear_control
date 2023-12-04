from visuals import video_to_frames
from utils import deserialize 

videos = deserialize("./data/videos.pkl")
video_to_frames(videos[0], "./frames")
