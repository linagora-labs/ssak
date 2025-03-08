#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(__file__))

from post_conversation import (
    cm_delete_conversation,
    cm_find_conversation,
    cm_get_tag_id,
)

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Remove conversations with a given tag in Studio (aka Conversation Manager).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tag", type=str, help="Tag")
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

    url, email, password = args.url, args.email, args.password

    tag_id = cm_get_tag_id(args.tag, url, email, password, create_if_missing=False, verbose=args.verbose)
    assert tag_id, "Tag not found"

    conversations = cm_find_conversation("", url, email, password, verbose=args.verbose)

    for conv in conversations:
        if tag_id in conv["tags"]:
            cm_delete_conversation(conv, url, email, password, verbose=args.verbose)
