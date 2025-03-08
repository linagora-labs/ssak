#!/usr/bin/env python3

import datetime
import json
import os
import re
import time

from ssak.utils.curl import curl_delete, curl_get, curl_post
from ssak.utils.format_diarization import to_linstt_diarization
from ssak.utils.format_transcription import shorten_transcription, to_linstt_transcription
from ssak.utils.linstt import linstt_transcribe
from ssak.utils.misc import hashmd5

####################
# Conversation Manager


def cm_import(
    audio_file,
    transcription,
    url,
    email,
    password,
    lang="fr-FR",
    name=None,
    tags=None,
    verbose=False,
):
    assert os.path.isfile(audio_file), f"File {audio_file} does not exist."
    assert isinstance(transcription, dict), f"Transcription must be a dict, got {type(transcription)}"
    if tags is None:
        tags = []

    if not name:
        name = os.path.splitext(os.path.basename(audio_file))[0] + " - UNK"

    token = cm_get_token(url, email, password, verbose=verbose)

    speakers = get_speakers(transcription)
    has_speaker = speakers != [None]
    has_punc = has_punctuation(transcription)

    datestr = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d - %H:%M:%S")

    organizationId = cm_get_organization(url, email, password, verbose=verbose)

    kargs = [
        url + f"/api/organizations/{organizationId}/conversations/import?type=transcription",
        {
            "transcription": transcription,
            "file": os.path.realpath(audio_file),
            "lang": lang,
            "name": name,
            "segmentCharSize": 2500,
            "transcriptionConfig": {
                "punctuationConfig": {"enablePunctuation": has_punc, "serviceName": "Custom" if has_punc else None},
                "diarizationConfig": {
                    "enableDiarization": has_speaker,
                    "numberOfSpeaker": len(speakers),
                    "maxNumberOfSpeaker": None,
                    "serviceName": "Custom" if has_speaker else None,
                },
                "enableNormalization": has_digit(transcription),
            },
            "description": f"Audio: {os.path.basename(audio_file)} / Transcription: {hashmd5(transcription)} / Import: {datestr}",
            "membersRight": "0",
        },
    ]
    kwargs = dict(
        headers=[f"Authorization: Bearer {token}"],
        verbose="short" if verbose else False,
    )
    result = curl_post(*kargs, **kwargs)

    # # The following is a failed trial of workaround, when curl inputs are too big. Using shell comand instead of pycurl does not work.
    # if result.get("message") == "Transcription is not a valid json":
    #     print("WARNING: pycurl failed, retrying with curl command line.")
    #     result = curl_post(*kargs, use_shell_command=True, **kwargs)

    assert "message" in result, f"'message' not found in response: {result}"
    assert result["message"] == "Conversation imported", f"Error when posting conversation: {result}"

    print("\n" + result["message"])

    if len(tags):
        conversation = cm_find_conversation(name, url, email, password, verbose=verbose, strict=True)
        assert len(conversation) > 0, f"Conversation not found: {conversation}"
        assert len(conversation) == 1, f"Multiple conversations found: {conversation}"
        conversation = conversation[0]
        conversationId = conversation["_id"]
    for tag in tags:
        tagId = cm_get_tag_id(tag, url, email, password, verbose=verbose)
        res = curl_post(
            url + f"/api/conversations/{conversationId}/tags/{tagId}",
            {
                "conversationId": conversationId,
                "tagId": tagId,
            },
            headers=[f"Authorization: Bearer {token}"],
            verbose=verbose,
        )
        # Previously: (res.get("status") == "OK" or res.get("message") == "Tag added to conversation")
        assert isinstance(res, dict) and (res.get("_id") is not None), f"Unexpected response: {res}"


def cm_get_organization(url, email, password, verbose=False):
    token = cm_get_token(url, email, password, verbose=verbose)
    organization = curl_get(
        url + "/api/organizations",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    assert len(organization) >= 1, "No organization found."
    if isinstance(organization, dict):
        assert "message" in organization
        raise RuntimeError(f"Error: {organization['message']}")
    organizationId = organization[0]["_id"]
    return organizationId


def cm_find_conversation(
    name,
    url,
    email,
    password,
    verbose=False,
    strict=False,
):
    token = cm_get_token(url, email, password, verbose=verbose)

    organization_id = cm_get_organization(url, email, password, verbose=verbose)

    conversations = curl_get(
        url + f"/api/organizations/{organization_id}/conversations",
        {"name": re.escape(name)},
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
        default={"conversations": []},
    )

    assert isinstance(conversations, dict) and "list" in conversations, f"'list' not found in response: {conversations}"
    conversations = conversations["list"]
    if strict:
        conversations = [c for c in conversations if c["name"] == name]

    return conversations


def cm_delete_conversation(
    conversation_id,
    url,
    email,
    password,
    verbose=False,
):
    if isinstance(conversation_id, dict):
        assert "_id" in conversation_id, f"'_id' not found in response: {conversation_id}"
        conversation_id = conversation_id["_id"]
    assert isinstance(conversation_id, str), f"Conversation ID must be a string, got {type(conversation_id)}"

    token = cm_get_token(url, email, password, verbose=verbose)

    result = curl_delete(
        url + f"/api/conversations/{conversation_id}/",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )

    assert "message" in result, f"'message' not found in response: {result}"
    if verbose:
        print(result["message"])
    return result


####################
# CM tags


def cm_get_tags(url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    token = cm_get_token(url, email, password, verbose=verbose)
    return curl_get(
        url + f"/api/organizations/{organizationId}/tags?categoryType=conversation_metadata",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )


def cm_get_categories(url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    token = cm_get_token(url, email, password, verbose=verbose)
    res = curl_get(
        url + f"/api/organizations/{organizationId}/categories",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    if isinstance(res, dict):
        assert "message" in res
        raise RuntimeError(f"Error: {res['message']}")
    assert isinstance(res, list)
    return [r["_id"] for r in res]


def cm_get_tag_id(tag, url, email, password, organizationId=None, create_if_missing=True, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)

    tags = cm_get_tags(url, email, password, organizationId=organizationId, verbose=verbose)
    for t in tags:
        if t["name"] == tag:
            return t["_id"]

    # The tag was not found
    if not create_if_missing:
        return None

    token = cm_get_token(url, email, password, verbose=verbose)
    categories = cm_get_categories(url, email, password, organizationId=organizationId, verbose=verbose)
    assert len(categories) > 0, "No tag category found."
    curl_post(
        url + f"/api/organizations/{organizationId}/tags",
        {
            "name": tag,
            "categoryId": categories[0],
        },
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    tags = cm_get_tags(url, email, password, organizationId=organizationId, verbose=verbose)
    for t in tags:
        if t["name"] == tag:
            return t["_id"]
    raise RuntimeError("There was an error: tag could not be created.")


def cm_remove_tag(tag, url, email, password, organizationId=None, verbose=False):
    if organizationId is None:
        organizationId = cm_get_organization(url, email, password, verbose=verbose)
    tagId = cm_get_tag_id(tag, url, email, password, organizationId=organizationId, create_if_missing=False, verbose=verbose)
    if tagId is None:
        raise RuntimeError(f"Tag {tag} does not exist.")
    token = cm_get_token(url, email, password, verbose=verbose)
    res = curl_delete(
        url + f"/api/organizations/{organizationId}/tags/{tagId}",
        headers=[f"Authorization: Bearer {token}"],
        verbose=verbose,
    )
    if res != "Tag deleted":
        raise RuntimeError(f"Error: {res}")


####################
# CM token

CM_TOKEN = {}


def cm_get_token(url, email, password, verbose=False, force=False):
    _id = f"{url}@{email}"

    global CM_TOKEN
    if _id in CM_TOKEN and not force:
        return CM_TOKEN[_id]

    token = curl_post(
        url + "/auth/login",
        {
            "email": email,
            "password": password,
        },
        verbose=verbose,
    )
    assert "auth_token" in token, f"'token' not found in response: {token}"
    CM_TOKEN[_id] = token = token["auth_token"]
    return token


####################
# Format conversion


def get_speakers(transcription):
    all_speakers = set()
    for seg in transcription["segments"]:
        all_speakers.add(seg["spk_id"])
    return list(all_speakers)


def has_punctuation(transcription):
    text = transcription["transcription_result"]
    for c in ".,;:?!":
        if c in text:
            return True
    return False


def has_digit(transcription):
    text = transcription["transcription_result"]
    for c in text:
        if c.isdigit():
            return True
    return False


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Post a conversation to Studio (aka Conversation Manager). Using https://alpha.linto.ai/cm-api/apidoc/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="Audio file")
    parser.add_argument("annotations", type=str, help="File with transcription or diarization results", default=None, nargs="?")
    parser.add_argument("-n", "--name", type=str, help="Name of the conversation", default=None)
    parser.add_argument("-t", "--tag", type=str, help="Tag for the conversation (or list of tags if seperated by commas)", default=None)
    parser.add_argument(
        "-e",
        "-u",
        "--email",
        "--username",
        type=str,
        help="Email of the Conversation Manager account (can also be passed with environment variable CM_EMAIL)",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        help="Password of the Conversation Manager account (can also be passed with environment variable CM_PASSWD)",
        default=None,
    )
    parser.add_argument("--url", type=str, help="Conversation Manager url", default="https://alpha.linto.ai")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing conversations with the same name")
    parser.add_argument("--new", action="store_true", help="Do not post if the conversation already exists")
    parser.add_argument(
        "--shorten_transcription",
        action="store_true",
        help="Shorten the transcription results, by using ellipsis (...) in long segments.",
    )

    parser_stt = parser.add_argument_group("Options to run transcription when no transcription is provided")
    parser_stt.add_argument(
        "--transcription_server",
        type=str,
        help="Transcription server",
        default="https://alpha.api.linto.ai/stt-french-whisper-v3",
        # default="https://alpha.api.linto.ai/stt-french-generic",
        # default="https://api.linto.ai/stt-french-generic",
    )
    parser_stt.add_argument("--num_spearkers", type=int, help="Number of speakers", default=None)
    parser_stt.add_argument("--convert_numbers", default=True, action="store_true", help="Convert numbers to text")
    parser_stt.add_argument("--diarization_service_name", default="stt-diarization-simple", help="Diarization service name")
    args = parser.parse_args()

    if not args.url.endswith("cm-api"):
        args.url = args.url.rstrip("/") + "/cm-api"

    if not args.email:
        args.email = os.environ.get("CM_EMAIL")
        if not args.email:
            raise ValueError("No CM email given. Please set CM_EMAIL environment variable, or use option -u.")

    if not args.password:
        args.password = os.environ.get("CM_PASSWD")
        if not args.password:
            raise ValueError("No CM password given. Please set CM_PASSWD environment variable, or use option -p.")

    base_filename = os.path.splitext(os.path.basename(args.audio))[0]
    default_name = base_filename

    annotations = args.annotations
    is_diarization = False
    if isinstance(annotations, str):
        extension = os.path.splitext(annotations)[1]
        if extension in [".rttm"]:
            is_diarization = True
        elif extension == ".json":
            try:
                with open(annotations, encoding="utf8") as f:
                    tmp = json.load(f)
                is_diarization = "speakers" in tmp
            except Exception as err:
                print(f"Error when reading annotations file: {err}")
                pass
    transcription = annotations if not is_diarization else None
    diarization = to_linstt_diarization(annotations, remove_overlaps=True) if is_diarization else None

    if not transcription:
        default_name += " | STT " + os.path.basename(args.transcription_server).replace("stt", "").strip(" _-")

        transcription = linstt_transcribe(
            args.audio,
            transcription_server=args.transcription_server,
            diarization=diarization if is_diarization else args.num_spearkers,
            convert_numbers=args.convert_numbers,
            diarization_service_name=args.diarization_service_name,
            verbose=args.verbose,
        )
        if args.verbose:
            print("\nTranscription results:")
            print(json.dumps(transcription, indent=2, ensure_ascii=False))

    else:
        if os.path.isfile(transcription):
            default_name += " | " + os.path.splitext(os.path.basename(transcription))[0].replace(base_filename, "").strip(" _-")
            with open(transcription, encoding="utf8") as f:
                try:
                    transcription = json.load(f)
                except Exception as err:
                    raise ValueError(f"Transcription file {transcription} is not a valid json file.") from err
        else:
            try:
                transcription = json.loads(transcription)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f"Transcription '{transcription[:100]}' : file not found, and not a valid json string.") from err

    if is_diarization:
        default_name += " | diar. " + os.path.splitext(os.path.basename(annotations))[0].replace(base_filename, "").strip(" _-")

    transcription = to_linstt_transcription(transcription, include_confidence=False)  # Confidence score not needed in LinTO Studio
    if args.shorten_transcription:
        transcription = shorten_transcription(transcription)

    name = args.name if args.name else default_name

    conversations = cm_find_conversation(
        name,
        url=args.url,
        email=args.email,
        password=args.password,
        verbose=args.verbose,
        strict=True,
    )
    if len(conversations):
        s = "s" if len(conversations) > 1 else ""
        names = ", ".join(list(set([conv["name"] for conv in conversations])))
        print(f"Already found {len(conversations)} conversation{s} with name{s}: '{names}'")
        if args.overwrite:
            assert not args.new
            x = "d"
        elif args.new:
            x = ""
        else:
            x = "_"
        while x.lower() not in ["", "i", "d"]:
            x = input("Do you want to ignore and continue (i), delete conversations and continue (d), or abort (default)?")
        if "i" in x.lower():
            pass
        elif "d" in x.lower():
            print("Delete other conversation.")
            for conv in conversations:
                cm_delete_conversation(
                    conv,
                    url=args.url,
                    email=args.email,
                    password=args.password,
                    verbose=args.verbose,
                )
        else:
            print("Abort.")
            sys.exit(0)

    cm_import(
        args.audio,
        transcription,
        name=name,
        tags=[t for t in args.tag.split(",") if t] if args.tag else [],
        url=args.url,
        email=args.email,
        password=args.password,
        verbose=args.verbose,
    )
