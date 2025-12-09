import csv
import logging
import os
import re

from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn
from ssak.utils.kaldi_converter import Row2KaldiInfo, DatasetProcessor2Kaldi

logger = logging.getLogger(__name__)

LOG_FOLDER = "nemo_data_conversion"

class Reader2Nemo:
    """
    Convert a dataset to NeMo format using a list of processors.
    The processors are executed in the order of the execute_order attribute of processors
    """
    def __init__(self, root, processors) -> None:
        for i in processors:
            if not isinstance(i, Row2KaldiInfo) and not isinstance(i, DatasetProcessor2Kaldi):
                if not os.path.exists(i.input):
                    i.input = os.path.join(root, i.input)
                    if not os.path.exists(i.input):
                        raise FileNotFoundError(f"File {i.input} not found")
        self.processors = processors

    def load(
        self,
        debug=False,
        dataset_name=None,
        custom_metadata_to_keep=None,
    ):
        if debug:
            logger.warning("Debug mode is on, will only process the first row")
        dataset = []
        self.processors = sorted(self.processors, key=lambda x: x.execute_order)
        pbar = tqdm(self.processors, desc="Processing pipeline")
        for processor in pbar:
            pbar.set_description(f"Processing {processor.__class__.__name__}")
            dataset = processor.process(dataset, debug=debug)
            if debug:
                logger.info(f"Step {processor.__class__.__name__}: {dataset}")
        logger.info(f"Dataset processed with {len(dataset)} rows")
        logger.info(f"First row: {dataset[0]}")
        nemo_dataset = NemoDataset(log_folder=LOG_FOLDER)
        # find the filters by finding all keys in first row that starts with "filter_"
        filters = [k for k in dataset[0] if k.startswith("filter_")]
        if len(filters) > 0:
            logger.info(f"Found filters: {filters}")
            filter_files = dict()
            for f in filters:
                filter_files[f] = open(os.path.join(LOG_FOLDER, f"{f}.txt"), "w")
        for row in tqdm(dataset, desc="Creating NeMo dataset"):
            if all(row[f] for f in filters):
                turns = []
                if row.get("context"):
                    turns.append(NemoTurn(role="User", value=row["context"], turn_type="text"))
                turns.append(NemoTurn(role="User", value=row["audio_path"], turn_type="audio", duration=row["duration"], offset=row.get("offset", 0.0)))
                turns.append(NemoTurn(role="Assistant", value=row["answer"], turn_type="text"))
                metadata = None
                if custom_metadata_to_keep:
                    metadata = {k: row[k] for k in custom_metadata_to_keep if k in row}
                nemo_row = NemoDatasetRow(
                    turns=turns,
                    id=row["id"],
                    dataset_name=row.get("dataset_name", dataset_name),
                    custom_metadata=metadata
                )
                nemo_dataset.append(nemo_row)
            else:
                for f in filters:
                    if not row[f]:
                        filter_files[f].write(f"{row}\n")
        logger.info(f"Removed {len(dataset)-len(nemo_dataset)} rows (from {len(dataset)} rows to {len(nemo_dataset)})")
        return nemo_dataset