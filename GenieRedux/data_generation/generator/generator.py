import functools
import multiprocessing
import os.path as osp
import os
import cv2
import json

from tqdm import tqdm
from generator.connector_base import BaseConnector

multiprocessing.set_start_method('spawn', force=True)

class DatasetFileStructure:
    INSTANCE_ID = "$INSTID$"
    FRAME_ID = "$FRAMEID$"
    SESSION_ID = "$SESSIONID$"
    FRAME_FMT = "%06d"
    INSTANCE_FMT = "%06d"
    SESSION_FMT = "%06d"

    def __init__(self, data_dpath=""):
        if data_dpath != "":
            os.makedirs(data_dpath, exist_ok=True)

        self.data_dpath = data_dpath

        self.actions_json_fname = "actions.json"
        self.info_json_fname = "info.json"
        self.observations_dname = ""

        self.observations_dpath = osp.join(data_dpath, self.observations_dname)
        self.__instance_dpath = osp.join(self.observations_dpath, self.INSTANCE_ID)
        self.__session_dpath = osp.join(self.__instance_dpath, self.SESSION_ID)

        self.__frame_fpath = osp.join(
            self.__session_dpath, "frames", f"{self.FRAME_ID}.jpg"
        )
        self.__actions_fpath = osp.join(self.__session_dpath, self.actions_json_fname)
        self.info_fpath = osp.join(self.data_dpath, self.info_json_fname)

    def get_instance_dpath(self, instance_id, make_dirs=False):
        dpath = self.__instance_dpath.replace(
            self.INSTANCE_ID, self.INSTANCE_FMT % instance_id
        )
        if make_dirs:
            os.makedirs(dpath, exist_ok=True)
        return dpath

    def get_session_dpath(self, instance_id, session_id, make_dirs=False):
        dpath = self.__session_dpath.replace(
            self.INSTANCE_ID, self.INSTANCE_FMT % instance_id
        ).replace(self.SESSION_ID, self.SESSION_FMT % session_id)
        if make_dirs:
            os.makedirs(dpath, exist_ok=True)
        return dpath

    def get_action_fpath(self, instance_id, session_id, make_dirs=False):
        fpath = self.__actions_fpath.replace(
            self.INSTANCE_ID, self.INSTANCE_FMT % instance_id
        ).replace(self.SESSION_ID, self.SESSION_FMT % session_id)
        if make_dirs:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
        return fpath

    def get_frame_fpath(self, instance_id, session_id, frame_id, make_dirs=False):
        fpath = (
            self.__frame_fpath.replace(
                self.INSTANCE_ID, self.INSTANCE_FMT % instance_id
            )
            .replace(self.FRAME_ID, self.FRAME_FMT % frame_id)
            .replace(self.SESSION_ID, self.SESSION_FMT % session_id)
        )
        if make_dirs:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
        return fpath


class EnvironmentDataGenerator:
    def __init__(
        self, connector_class_name, connector_config, generator_config, config
    ):

        self.connector_config = connector_config.copy()
        self.connector_class_name = connector_class_name
        env_connector: BaseConnector = connector_class_name(connector_config)

        # Set up the data directory file structure
        env_dname = env_connector.name + "_v" + env_connector.version
        dname = env_connector.get_name() + "_v" + env_connector.version
        if config["dname"] != "":
            env_dname += "_" + config["dname"]
            dname += "_" + config["dname"]
        self.name = dname
        self.data_dpath = osp.join(config["data_dpath"], env_dname, dname)
        self.fs: DatasetFileStructure = DatasetFileStructure(self.data_dpath)

        # Set up the dataset info
        self.info = {}
        self.info["info"] = env_connector.get_info()
        self.info["name"] = env_connector.get_name()
        self.info["generator_version"] = "0.1.0"
        self.info["version"] = env_connector.version

        # Set up the generator
        self.n_instances = generator_config["n_instances"]
        self.n_sessions = generator_config["n_sessions"]
        self.n_steps_max = generator_config["n_steps_max"]
        self.n_workers = generator_config["n_workers"]

    @staticmethod
    def generate_data(
        fs: DatasetFileStructure,
        connector_class_name,
        connector_config,
        n_steps_max,
        ids,
    ):
        env_connector = connector_class_name(connector_config)
        for instance_id, session_id in tqdm(ids):
            actions = []
            for data in env_connector.generator(
                instance_id=instance_id, session_id=session_id, n_steps_max=n_steps_max
            ):

                src_frame_id = data["src_frame_id"]
                tgt_frame_id = data["tgt_frame_id"]
                frame = data["frame"]
                action = data["action"]
                data["session_end"]
                extras = data["extras"]
                actions.append(
                    {
                        "src_id": src_frame_id,
                        "tgt_id": tgt_frame_id,
                        "action": action,
                        "extras": extras,
                    }
                )

                fs.get_instance_dpath(instance_id, make_dirs=True)
                fs.get_session_dpath(instance_id, session_id=session_id, make_dirs=True)

                frame_fpath = fs.get_frame_fpath(
                    instance_id, session_id, tgt_frame_id, make_dirs=True
                )

                cv2.imwrite(frame_fpath, frame[:, :, ::-1])

            actions_fpath = fs.get_action_fpath(
                instance_id, session_id=session_id, make_dirs=True
            )
            with open(actions_fpath, "w") as f:
                json.dump({"actions": actions}, f)

    def generate(self):
        os.makedirs(self.fs.data_dpath, exist_ok=True)
        with open(self.fs.info_fpath, "w") as f:
            json.dump(self.info, f)

        # Generate all pairs of instance and session ids based on self.n_sessions and self.n_instances
        worker_params = [
            (instance_id, session_id)
            for instance_id in range(self.n_instances)
            for session_id in range(self.n_sessions)
        ]
        # partition worker_params into n_workers partitions. the parameters per worker have to be sequential

        chunk_size = len(worker_params) // self.n_workers
        worker_params_chunks = [
            worker_params[i : i + chunk_size]
            for i in range(0, len(worker_params), chunk_size)
        ]
        assert sum([len(chunk) for chunk in worker_params_chunks]) == len(worker_params)
        pool = multiprocessing.Pool(self.n_workers)
        gen_data = functools.partial(
            EnvironmentDataGenerator.generate_data,
            self.fs,
            self.connector_class_name,
            self.connector_config,
            self.n_steps_max,
        )
        pool.map(gen_data, worker_params_chunks)

        pool.close()
        pool.join()
