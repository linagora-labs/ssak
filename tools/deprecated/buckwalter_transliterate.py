import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from lang_trans.arabic import buckwalter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def translate_chunk(chunk, encoding):
    if encoding == "bw":
        return buckwalter.transliterate(chunk)
    elif encoding == "utf8":
        return buckwalter.untransliterate(chunk)


def translate_file(input_file, output_file, encoding):
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        while True:
            chunk = f_in.read(1024)
            if not chunk:
                break
            translated_chunk = translate_chunk(chunk, encoding)
            f_out.write(translated_chunk)


def translate_folder(input_folder, output_folder, encoding, parallel=False):
    os.makedirs(output_folder, exist_ok=True)
    input_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    num_files = len(input_files)

    with ProcessPoolExecutor() as executor:
        futures = {}
        for i, file_name in enumerate(input_files):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_{encoding}.txt")
            if parallel:
                futures[executor.submit(translate_file, input_file, output_file, encoding)] = file_name
            else:
                logger.info(f"Translating file {i+1}/{num_files}: {file_name}")
                translate_file(input_file, output_file, encoding)

        if parallel:
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error translating file {file_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", help="An input file path or folder", type=str, required=True)
    parser.add_argument("-o", help="An output file path or folder", type=str, required=True)
    parser.add_argument("-e", help="Encoder should be bw or utf8", type=str, required=True)
    parser.add_argument("-p", help="Use parallel processing", default=False, action="store_true")
    args = parser.parse_args()

    input_path = args.i
    output_path = args.o
    encoding = args.e
    use_parallel = args.p

    if os.path.isfile(input_path):
        logger.info(f"Translating file: {input_path}")
        translate_file(input_path, output_path, encoding)
    elif os.path.isdir(input_path):
        logger.info(f"Translating folder: {input_path}")
        translate_folder(input_path, output_path, encoding, parallel=use_parallel)
    else:
        logger.error("Invalid input: must be a file or a folder")
