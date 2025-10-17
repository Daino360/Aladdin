import fcntl
from pathlib import Path

import cv2
from PIL import Image

from typing import Dict, Tuple, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T, transforms

from data_generation.data.data import (
    DEFAULT_VERSION,
    DatasetFileStructure,
    DatasetFileStructureInstance,
)

import math
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import json
import dill
import random
from packaging.version import Version
import pandas as pd

from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from tools.logger import getLogger
from tqdm import tqdm
import time

log = getLogger("data", name_color="blue")

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# image related helpers functions and dataset


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, exts=["jpg", "jpeg", "png"]):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        print(f"{len(self.paths)} training samples found at {folder}")

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(image_size),
                T.RandomHorizontalFlip(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif


CHANNELS_TO_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f"channels {channels} invalid"
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_pil_images(tensor, only_first_image=True):
    tensor = torch.clamp(tensor, min=0, max=1)  # clipping underflow and overflow

    if only_first_image:
        return T.ToPILImage()(tensor.unbind(dim=1)[0])

    # convert all images to PIL and concatenate them to a single PIL image
    return T.ToPILImage()(torch.cat([t for t in tensor.unbind(dim=1)], dim=2))


def video_tensor_to_gif(
    tensor, path, duration=120, loop=0, optimize=True, actions=None
):

    tensor = torch.clamp(tensor, min=0, max=1)  # clipping underflow and overflow
    images = map(T.ToPILImage(), tensor.unbind(dim=1))

    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        loop=loop,
        optimize=optimize,
        duration=80,
    )
    return images


# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def tqdm_function_decorator(total, *args, **kwargs):

    class PbarFunctionDecorator(object):

        def __init__(self, func):
            self.func = func
            self.pbar = tqdm(total=total, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            tmp = self.func(*args, **kwargs)
            self.pbar.update()
            return tmp

    return PbarFunctionDecorator


class DatasetFileStructureSessionLibrary:
    """A collection of all subject file structures in an environment dataset."""

    def __init__(self, root: str | Path, version=DEFAULT_VERSION, **kwargs) -> None:
        """Build a collection of all subject file structures in the Dreyeve dataset."""
        root = Path(root)
        self.fs = DatasetFileStructure(str(root), version=version, **kwargs)
        self.version = self.fs.version
        self.instance_ids = self.fs.get_instance_ids()
        self.sessions = {
            i: DatasetFileStructureInstance(root, i, self.fs.version, **kwargs) for i in self.instance_ids  # type: ignore
        }

    def __getitem__(self, key: int) -> DatasetFileStructureInstance:
        """Get a subject file structure by its ID."""
        return self.sessions[key]

    def __iter__(self):
        """Iterate over the subjects."""
        return iter(self.sessions.values())

    def __len__(self):
        """Return the number of subjects."""
        return len(self.sessions)


class DatasetOutputFormat:
    PVG = "pvg"
    IVG = "ivg"
    LAM = "lam"
    VICREG = "vicreg"
    GENERAL = "general"

class EnvironmentDataset:

    __DATASET_SPLIT = {
        "train": [0, 0.9],
        "validation": [0.9, 0.99],
        "test": [0.99, 1],
        "all": [0, 1],
    }

    def __init__(
        self,
        root_dpath,
        seq_length_input: int = 5,
        seq_step: int = 1,
        enable_test: bool = False,
        split: str = "train",
        split_type: str = "frame",
        clip_length: int = 0,
        max_size: int | None = None,
        format: str = DatasetOutputFormat.GENERAL,
        transform=None,
        img_ext="jpg",
        instance_filter: dict | None = None,
        enable_cache: bool = True,
        cache_dpath: str = "cache",
        occlusion_mask: np.ndarray | None = None,
    ) -> None:
        """
        Initializes the Data class.

        Args:
            root_dpath (str): The root directory path.
            seq_length_input (int, optional): The input sequence length. Defaults to 5.
            seq_step (int, optional): The sequence step. Defaults to 1.
            enable_test (bool, optional): Flag to enable test mode. Defaults to False.
            split (str, optional): The dataset split. Defaults to "train".
            split_type (str, optional): The split type. One of "instance" or "frame". Defaults to "frame".
            clip_length (int, optional): The clip length. If set to 0, the clip length is set to 1e12. Defaults to 0.
            format (str, optional): The dataset format. Defaults to DatasetOutputFormat.GENERAL.
            transform (optional): The data transformation function. Defaults to None.
            instance_filter (optional): A dict of acceptable session properties to use. Defaults to None.
        """

        self.seq_length_input = seq_length_input
        self.seq_length_variable = seq_length_input
        self.seq_step = seq_step
        self.enable_test = enable_test
        self.format = format
        self.transform = transform
        self.session_filter = instance_filter
        self.occlusion_mask = occlusion_mask
        self.split = split
        self.split_type = split_type
        self.clip_length = clip_length

        if self.occlusion_mask is not None:
            log.i("Using occlusion mask!")

        self.info = EnvironmentDataset.__read_info(
            DatasetFileStructure(root_dpath).info_fpath
        )
        self.name = (
            self.info["name"]
            if self.info["name"] != "default"
            else Path(root_dpath).name
        )
        self.actions_shape = self.info["action_space"]

        self.fsl = DatasetFileStructureSessionLibrary(
            root_dpath, version=Version(self.info["version"]), extension=img_ext
        )
        self.lock = Lock()
        self.data = {}

        # Set up cache
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache_dpath = Path(cache_dpath)
            self.cache_dpath.mkdir(parents=True, exist_ok=True)
            self.cache_metadata_fpath = (
                self.cache_dpath / f"metadata_{self.name}_{self.fsl.version}.pkl"
            )
            if self.clip_length > 0:
                self.cache_data_seq_fpath = (
                    self.cache_dpath
                    / f"dataseq_{self.split_type}_{self.__DATASET_SPLIT[self.split][0]}_{self.__DATASET_SPLIT[self.split][1]}_{self.name}_{self.fsl.version}_{self.seq_length_variable}_{self.seq_step}_{self.clip_length}.feather"
                )
            else:
                self.cache_data_seq_fpath = (
                    self.cache_dpath
                    / f"dataseq_{self.split_type}_{self.__DATASET_SPLIT[self.split][0]}_{self.__DATASET_SPLIT[self.split][1]}_{self.name}_{self.fsl.version}_{self.seq_length_variable}_{self.seq_step}.feather"
                )

        self.metadata = self.__create_metadata()
        self.n_instances = len(list(self.metadata.keys()))
        self.max_size = max_size
        self.split = split
        self.split_type = split_type

        self.metadata = self.__build_split(self.metadata, split, split_type)
        self.data = self.__create_data(self.seq_length_variable)

    @staticmethod
    def __read_info(info_fpath):

        if info_fpath.exists():
            with open(info_fpath, "r") as json_file:
                info = json.load(json_file)
        else:
            log.w(f"Info file not found at {info_fpath}. Using default info.")
            info = {"info": {}}

        default_info = {
            "action_space": [1],
            "observation_space": None,
            "version": f"{DEFAULT_VERSION}.0",
            "name": "default",
        }

        for key in default_info:
            if key not in info:
                info[key] = default_info[key]

        return info

    def __get_n_actions(self):
        # build a map of discrete actions
        if self.n_actions is None:
            actions = list(self.metadata.values())[0]["action"].to_numpy()
            # if isinstance(actions[0], list):
            # self.n_actions = len(actions[0])
            if isinstance(actions[0], list):
                self.n_actions = len(actions[0])
            else:
                self.n_actions = self.info["info"]["action_space"][-1]
            # else:
            #     unique_top_left_values = np.unique(actions)
            #     unique_top_left_values.sort()
            #     # self.n_actions = len(unique_top_left_values)
            #     self.n_actions = 7

        return self.n_actions

    def _one_hot(self, x, n):
        try:
            return np.eye(n)[np.array(x).flatten().astype(int)]
        except IndexError:
            return np.eye(n)[np.zeros_like(x).flatten().astype(int)]

    def __build_split(self, metadata, split, split_type="instance"):

        n_elements = len(metadata)

        if split_type == "instance":
            dataset_split = range(
                int(math.floor(n_elements * self.__DATASET_SPLIT[split][0])),
                int(math.floor(n_elements * self.__DATASET_SPLIT[split][1])),
            )

            metadata = {id: metadata[id] for id in dataset_split}
        elif split_type == "session":

            for i in range(n_elements):
                # get all unique values of session_id in the dataframe metadata[i]
                session_ids = sorted(list(metadata[i]["session_id"].unique()))
                session_ids = session_ids[
                    int(
                        math.floor(len(session_ids) * self.__DATASET_SPLIT[split][0])
                    ) : int(
                        math.floor(len(session_ids) * self.__DATASET_SPLIT[split][1])
                    )
                ]

                # select only those elements of metadata[i] which elements are in session_ids
                metadata[i] = metadata[i][metadata[i]["session_id"].isin(session_ids)]
        elif split_type == "frame":
            for i in range(n_elements):
                metadata[i] = metadata[i].iloc[
                    int(
                        math.floor(
                            len(metadata[i].index) * self.__DATASET_SPLIT[split][0]
                        )
                    ) : int(
                        math.floor(
                            len(metadata[i].index) * self.__DATASET_SPLIT[split][1]
                        )
                    )
                ]

        else:
            raise ValueError(f"Invalid split type: {split_type}")

        # #assert that the intervals in dataset_split are not intersecting
        # for split1 in dataset_split:
        #     for split2 in dataset_split:
        #         if split1 != split2:
        #             assert len(set(dataset_split[split1]).intersection(set(dataset_split[split2]))) == 0

        return metadata

    def __create_metadata(self):
        has_read_error = False
        if self.enable_cache and self.cache_metadata_fpath.exists():
            log.i("Loading metadata from cache... : ", self.cache_metadata_fpath)
            with open(self.cache_metadata_fpath, "rb") as f:
                try:
                    metadata = dill.load(f)
                    # metadata = {}
                    # metadata_from_file = pd.read_feather(f)
                    # for instance_id in metadata_from_file["instance_id"].unique():
                    #     metadata[instance_id] = metadata_from_file[
                    #         metadata_from_file["instance_id"] == instance_id
                    #     ]

                except EOFError as e:
                    log.e(
                        f"EOFError loading metadata from cache: {self.cache_metadata_fpath}"
                    )
                    has_read_error = True
                except OSError as e:
                    log.e(
                        f"OSError loading metadata from cache: {self.cache_metadata_fpath}"
                    )
                    has_read_error = True
                except Exception as e:
                    log.e(
                        f"Error loading metadata from cache: {self.cache_metadata_fpath}"
                    )
                    has_read_error = True
                finally:
                    pass
        if (
            not (self.enable_cache and self.cache_metadata_fpath.exists())
            or has_read_error
        ):
            metadata = {}
            for session in tqdm(self.fsl, disable=len(self.fsl) <= 150):
                # Rest of the code inside the loop

                sessions = []
                for key in session.get_session_ids():
                    with open(
                        session.get_action_fpath(
                            session.instance_id, session_id=int(key)
                        ),
                        "r",
                    ) as json_file:
                        actions = json.load(json_file)
                        if "actions" in actions:
                            actions = actions["actions"]

                    session_pd = pd.DataFrame(actions)
                    session_pd["session_id"] = int(key)
                    sessions.append(session_pd)
                metadata_entry = pd.concat(sessions, axis=0, ignore_index=True)

                src_id_tag = (
                    "src_frame_id"
                    if "src_frame_id" in metadata_entry.columns
                    else "src_id"
                )
                tgt_id_tag = (
                    "tgt_frame_id"
                    if "tgt_frame_id" in metadata_entry.columns
                    else "tgt_id"
                )

                # metadata_entry["action"] = "click"

                # delete any alements where src_id is -1
                metadata_entry = metadata_entry[metadata_entry[src_id_tag] != -1]
                metadata_entry = metadata_entry[metadata_entry[tgt_id_tag] != -1]
                metadata_entry["src_frame_id"] = metadata_entry[src_id_tag].astype(
                    np.int32
                )
                metadata_entry["tgt_frame_id"] = metadata_entry[tgt_id_tag].astype(
                    np.int32
                )
                metadata_entry["session_id"] = metadata_entry["session_id"].astype(
                    np.int32
                )
                # metadata_entry["is_winning"] = metadata_entry["is_winning"].astype(bool)

                metadata_entry = metadata_entry.sort_values(
                    by=["session_id", "src_frame_id"]
                )

                metadata[session.instance_id] = metadata_entry

            if len(metadata) == 0:
                raise ValueError("Empty dataset!")

            if self.enable_cache:
                with open(self.cache_metadata_fpath, "wb") as f:
                    try:
                        dill.dump(metadata, f)
                        # metadata_to_store = pd.concat(
                        #     [metadata[key].assign(instance_id=key) for key in metadata.keys()],
                        #     ignore_index=True,
                        # )
                        # pd.DataFrame(metadata_to_store).to_feather(f)
                    except EOFError as e:
                        log.e(
                            f"EOFError writing metadata to cache: {self.cache_metadata_fpath}"
                        )
                    except OSError as e:
                        log.e(
                            f"OSError writing metadata to cache: {self.cache_metadata_fpath}"
                        )
                    finally:
                        pass

        return metadata

    def __create_data(self, seq_length_variable, n_workers=4):
        self.seq_length_current = seq_length_variable
        self.n_actions = None

        if self.enable_cache:
            if self.clip_length > 0:
                self.cache_data_seq_fpath = (
                    self.cache_dpath
                    / f"dataseq_{self.split_type}_{self.__DATASET_SPLIT[self.split][0]}_{self.__DATASET_SPLIT[self.split][1]}_{self.name}_{self.fsl.version}_{self.seq_length_variable}_{self.seq_step}_{self.clip_length}.feather"
                )
            else:
                self.cache_data_seq_fpath = (
                    self.cache_dpath
                    / f"dataseq_{self.split_type}_{self.__DATASET_SPLIT[self.split][0]}_{self.__DATASET_SPLIT[self.split][1]}_{self.name}_{self.fsl.version}_{self.seq_length_variable}_{self.seq_step}.feather"
                )

        has_read_error = False
        if self.enable_cache and self.cache_data_seq_fpath.exists():
            log.i("Loading data from cache... : ", self.cache_data_seq_fpath)

            with open(self.cache_data_seq_fpath, "rb") as f:
                try:
                    data_feather = pd.read_feather(f)
                    data_feather["action"] = data_feather["action"].apply(
                        lambda entry: [list(sublist) for sublist in entry]
                    )
                    data = data_feather.to_dict(orient="records")
                    del data_feather
                except Exception as e:
                    log.e(f"Error loading data from cache: {self.cache_data_seq_fpath}")
                    has_read_error = True
                finally:
                    pass

            # turn the list of arrays stored in each element of data["action"] dataframe into a list of lists

            # data["action"] = [list(entry) for entry in data["action"]]

        if (
            not (self.enable_cache and self.cache_data_seq_fpath.exists())
            or has_read_error
        ):
            log.i("Creating data...")
            data = []
            seq_length = seq_length_variable

            if len(self.metadata) == 0:
                self.n_instances = 0
                raise ValueError(f"Empty dataset! : {self.name}")

            # build the data list
            @tqdm_function_decorator(
                total=math.ceil(len(self.metadata)), disable=len(self.metadata) <= 150
            )
            def process_metadata(
                metadata_key, metadata_entry, seq_length, seq_step, session_filter
            ):
                metadata_entry_groups = metadata_entry.groupby("session_id")
                if session_filter is not None:
                    metadata_entry_groups = [
                        (session_id, group)
                        for key in session_filter.keys()
                        for session_id, group in metadata_entry_groups
                        if group.iloc[0][key] in session_filter[key]
                    ]

                data = []
                for session_id, group in metadata_entry_groups:
                    first_start_id = 0 if self.fsl.fs.start_frame_fpath is None else 1
                    if self.clip_length > 0:
                        len_group = min(len(group), self.clip_length)
                    else:
                        len_group = len(group)

                    last_start_id = len_group - seq_length + 1
                    for i in range(first_start_id, last_start_id, seq_step):
                        seq_metadata_entry = group.iloc[i : i + seq_length, :]
                        data_entry = {
                            "env_instance_id": metadata_key,
                            "env_session_id": session_id,
                            "seq_start": i,
                            "seq_length": seq_length,
                            "src_frame_ids": seq_metadata_entry[
                                "src_frame_id"
                            ].tolist(),
                            "tgt_frame_ids": seq_metadata_entry[
                                "tgt_frame_id"
                            ].tolist(),
                            "action": seq_metadata_entry["action"].to_list(),
                            # "top_left": np.array(seq_metadata_entry["top_left"].to_list()),
                            # "bottom_right": np.array(seq_metadata_entry["bottom_right"].to_list()),
                            # "action_id": [unique_top_left_values.tolist().index(entry) for entry in seq_metadata_entry["action"].to_list()],
                            # "is_winning": seq_metadata_entry["is_winning"].iloc[0],
                        }
                        data.append(data_entry)

                return data

            # with ThreadPool(n_workers=4) as pool:
            with ThreadPool(processes=4) as pool:
                results = pool.starmap(
                    process_metadata,
                    [
                        (
                            metadata_key,
                            metadata_entry,
                            seq_length,
                            self.seq_step,
                            self.session_filter,
                        )
                        for metadata_key, metadata_entry in self.metadata.items()
                    ],
                    chunksize=500,
                )
                data = [entry for sublist in results for entry in sublist]

            if self.enable_cache:
                with open(self.cache_data_seq_fpath, "wb") as f:

                    try:
                        pd.DataFrame(data).to_feather(f)
                    except Exception as e:
                        log.e(
                            f"Error writing data to cache: {self.cache_data_seq_fpath}"
                        )
                        print(e)
                        raise e
                    finally:
                        pass

        return data

    def _read_frames(
        self,
        frame_fpath,
        start_frame_fpath,
        frame_ids,
        scaling_factor=1,
        transform=None,
    ):
        """Read frames from a folder."""

        def worker(frame_id, frame_fpath, scaling_factor=1, transform=None):
            """Read a single frame from a file."""
            # frame_id += 1 #TODO: fix frame_id+1
            frame = cv2.imread(str(frame_fpath).format(frame_id))
            if frame is None:
                raise ValueError(
                    f"Could not read frame {frame_id} from {str(frame_fpath).format(frame_id)}."
                )

            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * scaling_factor),
                    int(frame.shape[0] * scaling_factor),
                ),
                interpolation=cv2.INTER_AREA,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

            if self.occlusion_mask is not None:
                frame = frame * (1 - self.occlusion_mask) + self.occlusion_mask * 128

            if transform is not None:
                transformed_frame = transform(frame)
            else:
                transformed_frame = frame
            return transformed_frame

        max_workers = cpu_count()
        with ThreadPool(max_workers) as pool:
            frames = pool.starmap(
                worker,
                [
                    (
                        frame_id,
                        (
                            frame_fpath
                            if frame_id != 0 or start_frame_fpath is None
                            else start_frame_fpath
                        ),
                        scaling_factor,
                        transform,
                    )
                    for frame_id in frame_ids
                ],
            )

        frames = np.stack(frames, axis=0)
        if self.format == DatasetOutputFormat.PVG:
            frames = frames.transpose(0, 3, 1, 2)
        return frames

    def __getitem__(self, idx):

        # update data if seq_length_variable has changed, use locking
        if self.seq_length_variable != self.seq_length_current:
            with self.lock:
                if self.seq_length_variable != self.seq_length_current:
                    unique_str = (
                        f"{self.fsl.version}_{self.seq_length_input}_{self.seq_step}"
                    )
                    self.data = self.__create_data(self.seq_length_variable)
                    if idx >= len(self.data):
                        idx = idx % len(self.data)

        data = self.data[idx]
        instance_id = data["env_instance_id"]
        session_id = data["env_session_id"]
        src_frame_ids = data["src_frame_ids"][: self.seq_length_variable]
        tgt_frame_id = data["tgt_frame_ids"][self.seq_length_variable - 1]
        data["seq_start"]
        # seq_length = data["seq_length"]
        # top_left = data["top_left"][:self.seq_length_variable]
        # bottom_right = data["bottom_right"][:self.seq_length_variable]
        actions = data["action"][: self.seq_length_variable]

        start_frame_fpath = self.fsl[instance_id].start_frame_fpath
        frame_fpath = self.fsl[instance_id].get_frame_fpath(
            session_id=session_id, session_props=None, frame_id=None
        )

        frames = self._read_frames(
            frame_fpath,
            start_frame_fpath,
            src_frame_ids + [tgt_frame_id],
            scaling_factor=1,
            transform=(
                self.transform if self.format != DatasetOutputFormat.PVG else None
            ),
        )
        input_frames = frames[:-1]
        output_frame = frames[-1]

        actions = np.array([action for action in actions], dtype=np.uint8)
        if self.format == DatasetOutputFormat.IVG:
            is_first = np.zeros((self.seq_length_variable + 1, 1), dtype=int)
            is_first[0, :] = np.ones_like(is_first[0, :])
            if np.array(self.info["info"]["action_space"])[-1] > 1:
                if isinstance(
                    list(self.metadata.values())[0]["action"].to_numpy()[0], list
                ):
                    actions_one_hot = actions
                else:
                    actions = actions.reshape(-1, 1)
                    actions_one_hot = np.array(
                        [
                            self._one_hot(action, self.__get_n_actions())[0]
                            for action in actions
                        ]
                    )
            else:
                actions_one_hot = self._one_hot(actions, self.__get_n_actions())

            output = {
                "input_frames": np.concatenate(
                    [input_frames, np.array([output_frame])], axis=0
                ).astype(np.float32),
                "output_frame": np.array([output_frame]),
                "actions": np.array(
                    actions_one_hot, dtype=np.float32
                ),  # np.array(actions, dtype=int),
                "is_first": is_first,
                "instance_id": instance_id,  # np.array(self._one_hot(instance_id, self.n_instances), dtype=np.float32),
                "n_instances": self.n_instances,
                "src_frame_ids": np.array(src_frame_ids),
                "tgt_frame_id": np.array([tgt_frame_id]),
            }

        elif self.format == DatasetOutputFormat.VICREG:
            actions = actions.reshape(-1, 1)
            rand_id = random.randint(0, len(input_frames))
            random_image = (input_frames[1:] + [output_frame])[rand_id]

            output = {
                "input_frames": np.concatenate(
                    [input_frames[[0]], np.array([random_image])]
                ).astype(np.float32),
                "instance_id": np.array(
                    self._one_hot(instance_id, self.n_instances), dtype=np.float32
                ),
            }
        elif self.format == DatasetOutputFormat.LAM:
            output = np.concatenate(
                [input_frames, np.array([output_frame])], axis=0
            ).astype(np.float32)

        else:
            output = {
                "input_frames": np.array(input_frames),
                "output_frame": np.array([output_frame]),
                "actions": np.array(actions),
                "src_frame_ids": np.array(src_frame_ids),
                "tgt_frame_id": np.array([tgt_frame_id]),
            }

            if self.enable_test:
                output["frame_ids"] = src_frame_ids + [tgt_frame_id]

        return output

    def __len__(self):
        if self.max_size is not None and self.max_size < len(self.data):
            return self.max_size
        return len(self.data)

    def set_observations_count(self, observations_count):
        self.seq_length_variable = observations_count

    def sample_actions(self, forbidden_actions):
        actions = torch.zeros_like(forbidden_actions)
        n_seq = forbidden_actions.shape[1]
        n_actions = forbidden_actions.shape[2]
        for i in range(n_seq):
            forbidden_action = forbidden_actions[0, i]
            while True:
                action = self._one_hot(random.randint(0, n_seq - 1), n_actions)
                if forbidden_action != action:
                    break
            actions[0, i, :] = torch.from_numpy(action)

        return actions


class TransformsGenerator:

    @staticmethod
    def pad_to_match_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]):
        height, width = image.shape[:2]
        target_width, target_height = target_size

        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        # Determine padding
        if aspect_ratio > target_aspect_ratio:
            # Width is larger than target, pad top and bottom
            new_width = width
            new_height = int(width / target_aspect_ratio)
            top_pad = (new_height - height) // 2
            bottom_pad = new_height - height - top_pad
            pad_width = ((top_pad, bottom_pad), (0, 0), (0, 0))
        else:
            # Height is larger than target, pad left and right
            new_height = height
            new_width = int(height * target_aspect_ratio)
            left_pad = (new_width - width) // 2
            right_pad = new_width - width - left_pad
            pad_width = ((0, 0), (left_pad, right_pad), (0, 0))

        # Pad the image
        if len(image.shape) == 3:
            # Image has channels (e.g., RGB)
            padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)
        else:
            # Grayscale image
            pad_width = pad_width[:2]
            padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

        return padded_image

    @staticmethod
    def check_and_resize(
        target_crop: None | List[int], target_size: None | Tuple[int, int]
    ):
        """
        Creates a function that transforms input OpenCV images to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) tuple representing the target height and width
        :return: function that transforms an OpenCV image to the target size
        """

        # Creates the transformation function
        def transform(image: np.ndarray):
            if target_crop is not None:
                left, upper, right, lower = target_crop
                image = image[upper:lower, left:right]
            if target_size is not None and not all(
                dim == size for dim, size in zip(image.shape[:2], target_size)
            ):
                image = TransformsGenerator.pad_to_match_aspect_ratio(
                    image, target_size
                )
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def get_evaluation_transforms_config(config) -> Tuple:
        return TransformsGenerator.get_evaluation_transforms(
            config.data.crop, config.observation_space[config.enc_cnn_keys[0]]
        )

    @staticmethod
    def get_evaluation_transforms(crop_size, observation_space) -> Tuple:
        """
        Obtains the transformations to use for the evaluation scripts
        :param config: The evaluation configuration file
        :return: reference_transformation, generated transformation to use for the reference and the generated datasets
        """

        reference_resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )
        generated_resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )

        # Do not normalize data for evaluation
        reference_transform = transforms.Compose(
            [
                reference_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ]
        )
        generated_transform = transforms.Compose(
            [
                generated_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ]
        )

        return reference_transform, generated_transform

    @staticmethod
    def get_final_transforms_config(config) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """

        return TransformsGenerator.get_final_transforms(
            config.observation_space[config.encoder.enc_cnn_keys[0]][1:],
            config.data.crop,
        )

    @staticmethod
    def get_final_transforms(
        observation_space, crop_size
    ) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """
        # resize_transform = TransformsGenerator.check_and_resize(config.data.crop,
        #                                                         config.observation_space[config.encoder.enc_cnn_keys[0]][1:])

        resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )
        transform = transforms.Compose(
            [
                resize_transform,
                transforms.ToTensor(),
                # TransformsGenerator.to_float_tensor,
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }
