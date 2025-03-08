import io
import json
import os
import re
import tempfile
import urllib.parse

import certifi
import pycurl

####################
# Curl helpers

_json_dump_options = dict(
    separators=(",", ":"),  # Shorter string (no space)
    # ensure_ascii=False,
)


def shorten(s, maximum=500):
    if len(s) > maximum:
        return s[: maximum // 2] + "<<...>>" + s[-maximum // 2 :]
    return s


def format_option_for_curl(option, use_unicode=False, as_in_cmd=False, short=False, can_use_tempfile=False):
    if can_use_tempfile:
        option2 = use_tempfile_for_big_input(option)
        if option2 != option:
            print(f"WARNING: Using temporary file {option2}")
            return format_option_for_curl(
                option2,
                use_unicode=use_unicode,
                as_in_cmd=as_in_cmd,
                short=short,
                can_use_tempfile=False,
            )

    if isinstance(option, bool):
        return "true" if option else "false"
    if isinstance(option, dict):
        s = json.dumps(option, **_json_dump_options)
        return format_option_for_curl(s, use_unicode=use_unicode, as_in_cmd=as_in_cmd, short=short, can_use_tempfile=can_use_tempfile)
    if isinstance(option, str) and os.path.isfile(option):
        if as_in_cmd:
            return format_option_for_curl(f"@{option}", use_unicode=use_unicode)
        if use_unicode:
            option = option.encode("utf8")
        return (pycurl.FORM_FILE, option)
    if isinstance(option, str):
        if use_unicode:
            s = option.encode("utf8")
        else:
            s = option
        if short:
            return shorten(s)
        return s
    return format_option_for_curl(str(option), use_unicode)


# Hack to avoid "Argument list too long" error
# (which can have been the cause of the failure of pycurl in the first place)
_temporary_files = []


def use_tempfile_for_big_input(value, max_len=1_000_000):
    global _temporary_files
    temp_filename = None
    if isinstance(value, str) and len(value) > max_len:
        temp_filename = tempfile.mktemp(suffix=".txt")
        with open(temp_filename, "w") as f:
            f.write(value)
    elif isinstance(value, dict):
        str_value = json.dumps(value, **_json_dump_options)
        if len(str_value) > max_len:
            temp_filename = tempfile.mktemp(suffix=".json")
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
        value = str_value
    if temp_filename:
        print(f"INFO: Using tempfile {temp_filename}")
        _temporary_files.append(temp_filename)
        return temp_filename
    return value


def format_all_options_for_curl(options, **kwargs):
    return [(key, format_option_for_curl(value, **kwargs)) for key, value in (options.items() if isinstance(options, dict) else options)]


def curl_post(url, options, **kwargs):  # headers=[], post_as_fields=False, default=None, verbose=False):
    return _curl_do("POST", url, options=options, **kwargs)


def curl_get(url, options={}, **kwargs):
    return _curl_do("GET", url, options=options, **kwargs)


def curl_delete(url, **kwargs):
    return _curl_do("DELETE", url, options={}, **kwargs)


def _curl_do(
    action,
    url,
    options,
    headers=[],
    post_as_fields=False,
    default=None,
    verbose=False,
    use_shell_command=False,
    return_format="application/json",
):
    """
    Perform a curl request with the given action, url, options, headers, and post_as_fields flag.

    Parameters:
    - action: str
        The action to perform (GET, POST, DELETE)
    - url: str
        The url to perform the request on
    - options: dict or list of tuples
        The options to pass to the request
    - headers: list of str
        The headers to pass to the request
    - post_as_fields: bool
        Whether to pass the options
    - default: any
        The default value to return if the request fails
    - use_shell_command: bool
        Whether to use the shell command instead of the pycurl library
    - return_format: str
        The format to return the result in (application/json, text/plain, text/vtt, text/srt)
    """
    assert action in ["GET", "POST", "DELETE"], f"Unknown action {action}"

    use_pycurl = not use_shell_command
    c = pycurl.Curl() if use_pycurl else None

    c.setopt(pycurl.SSL_VERIFYPEER, 0)
    c.setopt(pycurl.SSL_VERIFYHOST, 0)

    # Example:
    # ("file", (c.FORM_FILE, "/home/jlouradour/data/audio/bonjour.wav")),
    # ("type", "audio/x-wav"),
    # ("timestamps", ""),
    # ("transcriptionConfig", json.dumps(transcription_config)),
    # ("force_sync", "false")

    try:
        can_use_tempfile = False  # This was a trial for a workaround, when curl inputs are too big. Replacing with a file does not work.
        if post_as_fields:
            options_curl = format_option_for_curl(options, use_unicode=(action != "GET"), can_use_tempfile=can_use_tempfile)
            options_curl2 = format_option_for_curl(options, use_unicode=False, as_in_cmd=True, short=(verbose == "short"))
        else:
            options_curl = format_all_options_for_curl(options, use_unicode=(action != "GET"), can_use_tempfile=can_use_tempfile)
            options_curl2 = format_all_options_for_curl(options, use_unicode=False, as_in_cmd=True, short=(verbose == "short"))
        options_str = ""

        if action == "GET":
            if use_pycurl:
                c.setopt(c.CAINFO, certifi.where())
            if len(options_curl):
                url += "?" + urllib.parse.urlencode(options_curl)
        if action == "DELETE":
            if use_pycurl:
                c.setopt(c.CUSTOMREQUEST, "DELETE")
            assert len(options_curl) == 0, "DELETE requests cannot have options"
        if use_pycurl:
            c.setopt(c.URL, url)
            c.setopt(c.HTTPHEADER, [f"accept: {return_format}"] + headers)  # ['Content-Type: multipart/form-data'] ?
        if action == "POST":
            if post_as_fields:
                if use_pycurl:
                    c.setopt(c.POSTFIELDS, options_curl)
                options_str = " \\\n\t".join([f"-d '{options_curl2}'"])
            else:
                if use_pycurl:
                    c.setopt(c.HTTPPOST, options_curl)
                else:
                    options_str_complete = " \\\n\t".join([f"-F '{key}={value}'" for key, value in options_curl])
                options_str = " \\\n\t".join([f"-F '{key}={value}'" for key, value in options_curl2])
        buffer = io.BytesIO()
        if use_pycurl:
            c.setopt(c.WRITEDATA, buffer)

        if verbose:
            headers_str = " \\\n\t".join([f"-H '{header}'" for header in headers]) + (" \\\n\t" if len(headers) else "")
            cmd_str = f"\ncurl -X '{action}' \\\n\t\
'{url}' \\\n\t\
-H 'accept: {return_format}' \\\n\t\
{headers_str}\
{options_str}".rstrip("\\\n\t ")
            # Do not print passwords
            cmd_str = re.sub(r"(-F 'password=b')([^']*)(')", r"\1XXX\3", cmd_str)
            print(cmd_str)

        if use_pycurl:
            c.perform()
            c.close()
            response_body = buffer.getvalue().decode("utf-8")
        else:
            headers_str = " \\\n\t".join([f"-H '{header}'" for header in headers]) + (" \\\n\t" if len(headers) else "")
            cmd_str = f"\ncurl -X '{action}' \\\n\t\
'{url}' \\\n\t\
-H 'accept: {return_format}' \\\n\t\
{headers_str}\
{options_str_complete}".rstrip("\\\n\t ")
            if verbose:
                print(cmd_str)
            # Get the stdout of the command
            response_body = os.popen(cmd_str).read()
            response_body = response_body.encode("utf-8")

        if not response_body and default:
            response_body = default
        else:
            if "json" in return_format:
                try:
                    response_body = json.loads(response_body)
                except json.decoder.JSONDecodeError:
                    if action != "DELETE":
                        raise RuntimeError(f"Curl request failed with:\n\t{response_body}")

    finally:
        global _temporary_files
        for temp_filename in _temporary_files:
            os.remove(temp_filename)

    return response_body
