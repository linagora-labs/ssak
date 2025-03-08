#!/usr/bin/env python3

import csv
import os
import re
import time
import traceback
import urllib.parse

import requests
from google_ngram_downloader import readline_google_store
from pytube import YouTube
from pytube.exceptions import AgeRestrictedError
from selenium import webdriver
from tqdm import tqdm
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable, YouTubeRequestFailed, YouTubeTranscriptApi

ALL_IDS = {}


def get_new_ids(video_ids, path, subpath, ignore_discarded=False):
    if path in ALL_IDS:
        all_video_ids = ALL_IDS[path]
    else:
        all_video_ids = []
        if subpath and os.path.isdir(f"{path}/{subpath}"):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f"{path}/{subpath}")]
        if not ignore_discarded and os.path.isdir(f"{path}/discarded"):
            all_video_ids += [os.path.splitext(f)[0] for f in os.listdir(f"{path}/discarded")]
        ALL_IDS[path] = all_video_ids
    res = [id for id in video_ids if id not in all_video_ids]
    for id in res:
        all_video_ids.append(id)
    return res


# can detect the language of text or transcription in any language
# def is_language(text, l='ar'):
#     lang, score = langid.classify(text)
#     return lang == l


def register_discarded_id(video_id, path, reason=""):
    path = f"{path}/discarded"
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{video_id}.txt", "w") as f:
        f.write(str(reason) + "\n")


def unregister_discarded_id(video_id, path):
    path = f"{path}/discarded"
    if os.path.isfile(f"{path}/{video_id}.txt"):
        print(f"Video {video_id} now accepted :)")
        os.remove(f"{path}/{video_id}.txt")


def is_automatic(language):
    # Check if the language string contains the word "auto" or the Arabic word "تلقائيًا"
    if "auto" in language.lower() or "تلقائيًا" in language:
        return True
    return False


def norm_language_code(language_code):
    if len(language_code) > 2 and "-" in language_code:
        return language_code.split("-")[0]
    return language_code


ERROR_WHEN_NOT_AVAILABLE = "Subtitles disabled for video"
ERROR_PROXY = "Proxy error"


def get_transcripts_if(vid, if_lang="fr", proxy=None, all_auto=False, verbose=True, max_retrial=float("inf")):
    try:
        if proxy:
            max_retrial = min(max_retrial, 0)
            transcripts = list(YouTubeTranscriptApi.list_transcripts(vid, proxies=http_proxies(proxy)))
        else:
            transcripts = list(YouTubeTranscriptApi.list_transcripts(vid))
    except TranscriptsDisabled:
        msg = f"{ERROR_WHEN_NOT_AVAILABLE} {vid}"
        if verbose:
            print(msg)
        return msg
    except VideoUnavailable as err:
        msg = str(err)
        if verbose:
            print(msg)
        return msg
    except requests.exceptions.ProxyError as err:
        msg = f"{ERROR_PROXY}: {err}"
        if verbose:
            print(msg)
        return msg
    # except Exception as e: # (requests.exceptions.HTTPError) as e:
    except (YouTubeRequestFailed, requests.exceptions.HTTPError, requests.exceptions.ChunkedEncodingError) as err:
        # The most common error here is "Too many requests" (because the YouTube API is rate-limited)
        # We don't catch a specific exception because scraping script should seldom fail
        # This could cause an infinite loop if the error always occurs, but then it should print a message every 2 minutes
        if max_retrial <= 0:
            msg = f"{ERROR_PROXY}: {err}"
            if verbose:
                print(msg)
            return msg
        else:
            print("WARNING: Error", type(err), str(err))
            print("Waiting 120 seconds...")
            time.sleep(120)
            return get_transcripts_if(vid, if_lang=if_lang, proxy=proxy, all_auto=all_auto, verbose=verbose, max_retrial=max_retrial - 1)
    except Exception as err:
        msg = f"{ERROR_PROXY}: {err}"
        if verbose:
            print(msg)
        return msg

    has_auto = max([norm_language_code(t.language_code) == if_lang and is_automatic(t.language) for t in transcripts])
    has_language = max([norm_language_code(t.language_code) == if_lang and (all_auto or not is_automatic(t.language)) for t in transcripts])
    only_has_language = has_language and len(transcripts) == 1
    discarded = not has_language
    if not all_auto:
        discarded = discarded or (not has_auto and not only_has_language)
    if not if_lang:
        discarded = False
    if discarded:
        msg = f"Video {vid} discarded. Languages: {', '.join(t.language for t in transcripts)}"
        if verbose:
            print(msg)
        return msg

    res = {}
    try:
        for t in transcripts:
            language_code = language = "???"
            language_code = t.language_code
            language = t.language
            lang = norm_language_code(language_code) if not is_automatic(language) else norm_language_code(language_code) + "_auto"
            # This can fail with "xml.etree.ElementTree.ParseError: not well-formed (invalid token): line XXX, column XXX"
            res[lang] = t.fetch()

    except Exception:
        msg = f"Video {vid} discarded. Error when getting transcript:\n{traceback.format_exc()}"
        print(msg)
        return msg
    return res


DRIVER = None


# scrape the ids using a search query
def search_videos_ids(search_query, open_browser=False, use_global_driver=True):
    global DRIVER
    if DRIVER is None or not use_global_driver:
        # Set up Firefox driver
        options = webdriver.FirefoxOptions()
        if not open_browser:
            options.add_argument("--headless")

        try:
            DRIVER = webdriver.Firefox(options=options)
        except Exception as err:
            raise RuntimeError(
                "Could not start Firefox driver. You may need to install Firefox:\n\
                sudo add-apt-repository ppa:mozillateam/ppa\n\
                sudo apt-get update\n\
                apt-get install -y --no-install-recommends firefox-esr"
            ) from err
    try:
        # Navigate to YouTube and search for videos with subtitles
        DRIVER.get("https://www.youtube.com/results?search_query=" + urllib.parse.quote(search_query) + "&sp=EgIQAQ%253D%253D")

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
        num_scrolls = 0
        while True:
            DRIVER.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            num_scrolls += 1
            print(f"Scrolled {num_scrolls} time{'s' if num_scrolls>1 else ''} for query \"{search_query}\"...")

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(DRIVER.page_source)))))
        print(f'Found {len(video_ids)} video IDs for query "{search_query}"')

    finally:
        if not use_global_driver:
            DRIVER.close()
    return video_ids


# scrape the ids using a search query with name of Channels
def search_videos_ids_from_channels(channel_name, open_browser=False, use_global_driver=True):
    global DRIVER
    if DRIVER is None or not use_global_driver:
        # Set up Firefox driver
        options = webdriver.FirefoxOptions()
        if not open_browser:
            options.add_argument("--headless")

        try:
            DRIVER = webdriver.Firefox(options=options)
        except Exception as err:
            raise RuntimeError(
                "Could not start Firefox driver. You may need to install Firefox:\n\
                sudo add-apt-repository ppa:mozillateam/ppa\n\
                sudo apt-get update\n\
                apt-get install -y --no-install-recommends firefox-esr"
            ) from err
    try:
        # Navigate to YouTube and search for videos with subtitles
        channel_video_url = "https://www.youtube.com/@" + urllib.parse.quote(channel_name) + "/videos"
        DRIVER.get(channel_video_url)

        # Click "ACCEPT ALL" button if it exists
        click_button(
            DRIVER,
            **{"class": "VfPpkd-LgbsSe VfPpkd-LgbsSe-OWXEXe-k8QpJ VfPpkd-LgbsSe-OWXEXe-dgl2Hf nCP5yc AjY5Oe DuMIQc LQeN7 IIdkle"},
        )

        # Scroll down the page to load more videos
        SCROLL_PAUSE_TIME = 2
        last_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
        num_scrolls = 0
        while True:
            DRIVER.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = DRIVER.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            num_scrolls += 1
            print(f"Scrolled {num_scrolls} time{'s' if num_scrolls>1 else ''} for query \"{channel_name}\"...")

        # Extract video IDs from search results
        video_ids = sorted(list(set(re.findall('"videoId":"([^"]{11})"', str(DRIVER.page_source)))))
        print(f"Found {len(video_ids)} video IDs on {channel_video_url}")

    finally:
        if not use_global_driver:
            DRIVER.close()
    return video_ids


def extract_audio_yt(vid, output_audio_dir, skip_if_exists=True, verbose=True):
    output_audio_dir = f"{output_audio_dir}/mp4"
    if not os.path.isdir(output_audio_dir):
        os.makedirs(output_audio_dir)
    output_video_file = f"{output_audio_dir}/{vid}.mp3"
    if skip_if_exists and os.path.isfile(output_video_file):
        if verbose:
            print(f"Video {vid} skipped (already extracted)")

    try:
        video = YouTube(f"https://www.youtube.com/watch?v={vid}")
        if video.age_restricted:
            if verbose:
                print(f"Video {vid} is age-restricted. Skipping extraction.")

        isok = False
        for _ in range(3):
            try:
                stream = video.streams.filter(only_audio=True).first()
            except (AttributeError, KeyError):
                import traceback

                traceback.print_exc()
                print(f"WARNING: got an error trying to extract the video {vid}. Retrying...")
                time.sleep(1)
                continue
            file_tmp = stream.download(output_path=output_audio_dir)
            file_withid = f"{output_audio_dir}/{vid}{os.path.splitext(file_tmp)[1]}"
            if file_tmp != file_withid:
                os.rename(file_tmp, file_withid)
            isok = True
            break

        if not isok:
            print(f"WARNING: could not get video {vid}")

    except AgeRestrictedError:
        print(f"WARNING: Video {vid} is age-restricted and cannot be accessed without logging in.")


def scrape_transcriptions(
    video_ids,
    path,
    if_lang,
    all_auto=False,
    skip_if_exists=True,
    skip_if_discarded=True,
    proxies=None,
    max_proxies=None,
    extract_audio=False,
    verbose=True,
):
    # Save videos_ids in a file
    n = len(video_ids)
    if skip_if_exists:
        video_ids = get_new_ids(video_ids, path, if_lang, ignore_discarded=not skip_if_discarded)
    print(f"Got {len(video_ids)} new video ids / {n}")

    for vid in tqdm(video_ids):
        has_been_register_as_failed = False
        num_proxies_tried = 0

        for proxy in get_proxies_generator(proxies):
            if proxy and max_proxies and num_proxies_tried >= max_proxies:
                break

            if check_proxy(proxy):
                if proxy:
                    num_proxies_tried += 1
                    if verbose:
                        print(f"Using proxy {proxy} ({num_proxies_tried}/{max_proxies})")

                # Get transcription
                transcripts = get_transcripts_if(vid, if_lang=if_lang, proxy=proxy, all_auto=all_auto, verbose=verbose)

                # If no transcription is available
                if not isinstance(transcripts, dict) or not transcripts:
                    if not has_been_register_as_failed:
                        register_discarded_id(vid, path, reason=transcripts)
                        has_been_register_as_failed = True
                    if isinstance(transcripts, str) and not transcripts.startswith(ERROR_WHEN_NOT_AVAILABLE) and not transcripts.startswith(ERROR_PROXY):
                        break

                    # if transcripts.startswith(ERROR_PROXY):
                    #     # if proxies == "auto":
                    #     #     CHECKED_PROXIES[proxy] = False
                    #     # Recheck proxy status
                    #     check_proxy(proxy, force=True)

                    # Continue trying with other proxies (?) if not a transcription
                    continue

                if not skip_if_exists or has_been_register_as_failed:
                    unregister_discarded_id(vid, path)

                if verbose:
                    print(f"Video {vid} accepted. Languages: {', '.join(transcripts.keys())}")

                for lan, transcript in transcripts.items():
                    output_dir = f"{path}/{lan}"
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    output_file = f"{output_dir}/{vid}.csv"
                    with open(output_file, "w") as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=";")
                        # Add header
                        csvwriter.writerow(["text", "start", "duration"])
                        # Write content
                        for line in transcript:
                            csvwriter.writerow([line["text"].replace("\n", " "), line["start"], line["duration"]])
                if extract_audio:
                    extract_audio_yt(vid, path, skip_if_exists=skip_if_exists, verbose=verbose)

                break


def generate_ngram(n, lan, min_match_count=10000, index_start=None):
    lang = {
        "en": "eng",
        "fr": "fre",
        "ru": "rus",
        "ar": "ara",
    }.get(lan)
    if not lang:
        try:
            from iso639 import languages

            lang = languages.get(part1=lan).part2b
        except ImportError:
            raise ValueError(f"Unknown language {lan}.\nCan be solved by running 'pip install iso-639'")
    current_word = None

    # Make all possible 2-grams of letters
    # Note: this should be changed for languages with different alphabet (e.g. Arabic, Chinese, ...)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    if n > 1:
        letters = [l1 + l2 for l1 in letters for l2 in letters]
    if index_start:
        letters = [l for l in letters if l >= index_start]

    for fname, url, records in readline_google_store(ngram_len=n, lang=lang, indices=letters):
        for record in records:
            if record.match_count < min_match_count:
                continue
            if record.ngram == current_word:
                continue
            current_word = record.ngram
            text = record.ngram
            text = re.sub("_[^ ]*", "", text)
            text = re.sub(" +", " ", text).strip()
            if re.match("^[a-zA-Z]", text):
                yield text


def robust_generate_ngram(n, lan, min_match_count=10000, index_start=None):
    """
        Avoid this kind of errors:

    Traceback (most recent call last):
      File "/home/jlouradour/src/stt-end2end-expes_hedi/tools/scraping/scrape_youtube.py", line 189, in <module>
        for query in queries:
      File "/home/jlouradour/src/stt-end2end-expes_hedi/tools/scraping/scrape_youtube.py", line 156, in generate_ngram
        for record in records:
      File "/usr/local/lib/python3.9/site-packages/google_ngram_downloader/util.py", line 53, in lines
        assert not last
    AssertionError
    """

    if isinstance(n, list):
        for i in n:
            for text in robust_generate_ngram(i, lan, min_match_count=min_match_count, index_start=index_start):
                yield text

    current_index_start = ""
    try:
        for text in generate_ngram(n, lan, min_match_count=min_match_count, index_start=index_start):
            current_index_start = max(current_index_start, text[:2].lower())
            yield text
    except (AssertionError, requests.exceptions.ChunkedEncodingError):
        print(traceback.format_exc())
        print("WARNING: Google NGram failed. Retrying...")
        for text in robust_generate_ngram(n, lan, min_match_count=min_match_count, index_start=current_index_start):
            yield text


# to generate ngram from file or more then one
def parse_ngrams(path, ns):
    if isinstance(ns, int):
        ns = [ns]
    ns = sorted(set(ns))
    if os.path.isfile(path):
        # If path is a file, parse ngrams from the file
        with open(path) as f:
            for line in f:
                words = line.strip().split()
                for i in ns:
                    for j in range(len(words) - i + 1):
                        yield " ".join(words[j : j + i])
    elif os.path.isdir(path):
        # If path is a directory, parse ngrams from all files in the directory
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            for ngram in parse_ngrams(filepath, ns=ns):
                yield ngram
    else:
        print("Invalid path:", path)


def click_button(driver, *kargs, verbose=True, max_trial=5, ignore_failure=True, **kwargs):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    button = ",".join(["@" + k for k in kargs]) + ",".join([f'@{k.rstrip("_")}="{v}"' for k, v in kwargs.items()])
    if verbose > 1:
        print("* Click on:", button)
    for itrial in range(max_trial):
        try:
            return WebDriverWait(driver, 0).until(EC.element_to_be_clickable((By.XPATH, f"//*[{button}]"))).click()
        except Exception as err:
            if itrial == max_trial - 1:
                print(err)
                if ignore_failure:
                    return
                else:
                    raise err
            time.sleep(0.5)


CHECKED_PROXIES = {}


def check_proxy(proxy, timeout=3, verbose=True, force=False):
    global CHECKED_PROXIES
    if not proxy:
        return True
    if proxy in CHECKED_PROXIES and not force:
        return CHECKED_PROXIES[proxy]
    res = False
    try:
        if verbose:
            print("Checking proxy", proxy)
        response = requests.get("https://www.youtube.com", proxies=http_proxies(proxy), timeout=timeout)
        if response.status_code == 200:
            res = True
        elif verbose > 2:
            print("Error type1:", response.status_code, response)
    except requests.exceptions.RequestException as err:
        if verbose > 2:
            print("Error type2:", type(err), str(err))
        if verbose:
            print(f"Rejecting {proxy}")
    CHECKED_PROXIES[proxy] = res
    return res


PROXY_REGISTERED_PROVIDERS = None
USE_FREE_PROXY = True


def auto_proxy():
    global CHECKED_PROXIES, PROXY_REGISTERED_PROVIDERS

    if USE_FREE_PROXY:
        from fp.fp import FreeProxy
    else:
        from proxy_randomizer import RegisteredProviders

        if PROXY_REGISTERED_PROVIDERS is None:
            PROXY_REGISTERED_PROVIDERS = RegisteredProviders()
            PROXY_REGISTERED_PROVIDERS.parse_providers()

    # First try without proxy
    yield None

    # Test the list of successful proxies
    for proxy in CHECKED_PROXIES:
        if CHECKED_PROXIES[proxy]:
            yield proxy

    # Try new ones
    while True:
        if USE_FREE_PROXY:
            yield str(FreeProxy().get()).split("http://")[-1]
        else:
            yield str(PROXY_REGISTERED_PROVIDERS.get_random_proxy())


def get_proxies_generator(proxies):
    if proxies == "auto":
        return auto_proxy()
    if not proxies:
        return [None]
    assert isinstance(proxies, list)
    return [None] + proxies


def http_proxies(proxy):
    return {"http": proxy.split()[0], "https": proxy.split()[0]}


if __name__ == "__main__":
    import argparse
    import os

    from ssak.utils.misc import hashmd5

    parser = argparse.ArgumentParser(
        description="Scrape YouTube subtitled videos in a given language, extracting transcripts, and possibly audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Output folder path where audio and annotations will be saved (default: YouTubeFr, or YouTubeLang for another language than French).",
        type=str,
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="The language code of the transcripts you want to retrieve. For example, 'en' for English, 'fr' for French, etc.",
        type=str,
    )
    parser.add_argument(
        "--extract_audio",
        default=False,
        action="store_true",
        help="If set, the audio will be downloaded (in mp4 format) and saved on the fly.",
    )
    parser.add_argument(
        "--all_auto",
        help="Extract Youtube content as soon as there is the language in the target language",
        action="store_true",
        default=False,
    )
    parser.add_argument("--video_ids", help="An explicit list of video ids.", type=str, default=None)
    parser.add_argument(
        "--search_query",
        help="The search query that you want to use to search for YouTube videos. If neither --search_query nor --video_ids are specified, a series of queries will be generated automatically.",
        type=str,
    )
    parser.add_argument("--search_channels", default=False, action="store_true", help="Whether to search for channels.")
    parser.add_argument(
        "--ngram",
        default="3",
        type=str,
        help="n-gram to generate queries (integer or list of integers separated by commas).",
    )
    parser.add_argument(
        "--query_index_start",
        help="If neither --search_query nor --video_ids are specified this is the first letters for the generated queries",
        type=str,
    )
    parser.add_argument("--open_browser", default=False, action="store_true", help="Whether to open browser.")
    parser.add_argument(
        "--proxies",
        default=None,
        help="Specify a file contain a list of proxies or a server to use (e.g., 52.234.17.87:80).",
    )
    parser.add_argument("--max_proxies", default=None, type=int, help="Maximum proxies per file")
    args = parser.parse_args()
    should_extract_audio = args.extract_audio

    args.ngram = [int(n) for n in args.ngram.split(",")]

    lang = args.language
    if not args.search_query and not args.video_ids:
        queries = robust_generate_ngram(args.ngram, lang, index_start=args.query_index_start)
    elif args.search_query is not None and (os.path.isdir(args.search_query) or os.path.isfile(args.search_query)):
        queries = parse_ngrams(args.search_query, ns=1 if args.search_channels else args.ngram)
    else:
        queries = [args.search_query] if args.search_query else [None]

    skip_if_discarded = True
    if args.video_ids:
        assert queries == [None], "Cannot provide both a search query and a list of video ids"
        skip_if_discarded = False

    path = args.path
    if not path:
        if lang:
            # YouTubeEn, YouTubeFr, etc.
            path = f"YouTube{lang[0].upper()}{lang[1:].lower()}"
        else:
            path = "YouTube"

    # proxy setup
    proxies = args.proxies
    if proxies:
        if os.path.isfile(proxies):
            with open(proxies) as f:
                proxies = [line.strip() for line in f]
        elif os.path.isdir(proxies):
            proxies = [os.path.splitext(f)[0] for f in os.listdir(proxies)]
        elif proxies.lower() == "auto":
            proxies = "auto"
        else:
            proxies = proxies.split(",")

    # Set up the API client
    for query in queries:
        if query:
            # Log to avoid doing twice the same query
            query = query.strip()
            if not args.search_channels:
                lockfile = f"{path}/queries/{hashmd5(query)}"
                if os.path.isfile(lockfile):
                    print(f'Skipping (already done) query "{query}"')
                    continue

                os.makedirs(f"{path}/queries", exist_ok=True)
                with open(lockfile, "w", encoding="utf8") as f:
                    f.write(query + "\n")

        try:
            isok = False

            if args.video_ids:
                assert query is None, "--search_query should not be specified when --video_ids is specified"
                if os.path.isfile(args.video_ids):
                    with open(args.video_ids) as f:
                        video_ids = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f]
                elif os.path.isdir(args.video_ids):
                    video_ids = [os.path.splitext(f)[0] for f in os.listdir(args.video_ids)]
                else:
                    video_ids = args.video_ids.split(",")
            elif args.search_channels:
                assert query is not None
                print(f'========== get videos id from channels for query: "{query}" =========')
                video_ids = search_videos_ids_from_channels(query, open_browser=args.open_browser)
            else:
                assert query is not None
                print(f'========== get videos id for query: "{query}" =========')
                video_ids = search_videos_ids(query, open_browser=args.open_browser)

            print(f"========== get subtitles for videos in {lang} =========")
            scrape_transcriptions(
                video_ids,
                path,
                lang,
                all_auto=args.all_auto,
                skip_if_discarded=skip_if_discarded,
                extract_audio=should_extract_audio,
                proxies=proxies,
                max_proxies=args.max_proxies,
            )

            isok = True

        finally:
            if query:
                if isok:
                    with open(f"{path}/queries/all.txt", "a") as f:
                        f.write(query + "\n")
                else:
                    os.remove(lockfile)
