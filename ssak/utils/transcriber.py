import os
import random
import re

import xmltodict

from ssak.utils.text_basic import collapse_whitespace, transliterate


def read_transcriber(
    trs_file,
    anonymization_level=0,
    remove_extra_speech=True,
    do_format_speaker_name=True,
    capitalize=True,
    replacements=None,
    verbose=True,
):
    basename = os.path.basename(trs_file.split(".")[0])
    basename = basename.lower()

    with open(trs_file, encoding=file_encoding(trs_file)) as f:
        file = f.read()
        header, rest = file.split("<Trans", 1)
        file = header.strip() + "\n<Trans" + rest.rstrip().replace("\n\n", "\n" + HACK_EMPTY_LINES + "\n")
        file = file.splitlines()

    # Split lines according to hypothesis made in preformatXML...
    def split_line(line):
        line = re.sub(r'(<Who nb=".*"/>[^<]+) *<Sync time="[0-9\.]+"/>', r"\1", line)
        # Split so that <*> gives raise to a new line
        return [l.strip() for l in re.sub("(</?[^>]*>)", r"\n\1\n", line).split("\n") if len(l.strip()) > 0]

    file = [item for sublist in list(map(split_line, file)) for item in sublist]

    file = preformatXML(file, remove_extra_speech=remove_extra_speech)

    # For debug...
    # debug_file = "tmp_"+os.path.splitext(os.path.basename(trs_file))[0]+".xml"
    # print("DEBUG FILE: ", debug_file)
    # with open(debug_file, "w") as f:
    #     f.write(file)

    dict = xmltodict.parse(file)

    # prepare the list of speakers
    newSpkId = 1
    speaker_id = []
    speaker_gender = []
    speaker_name = []
    speaker_scope = []
    alldata = []

    if "Speakers" in dict["Trans"] and dict["Trans"]["Speakers"] is not None:
        speakers = dict["Trans"]["Speakers"]["Speaker"]
        if "@id" in dict["Trans"]["Speakers"]["Speaker"]:
            speakers = [dict["Trans"]["Speakers"]["Speaker"]]
        for spk in speakers:
            speaker_id.append(spk["@id"])
            speaker_gender.append(spk["@type"].lower()) if "@type" in spk else speaker_gender.append("unknown")
            spkname = spk["@name"] if "@name" in spk else ""
            if not spkname and "@id" in spk:
                spkname = spk["@id"]
            if not spkname:
                print("WARNING: missing speaker name")
                spkname = "unknown"
            else:
                spkname = format_speaker_name(spkname, strong=do_format_speaker_name)
            # Replace question marks by random numbers
            spkname = re.sub(r"\?", lambda x: "-" + str(random.randint(0, 90)), spkname)
            if spkname.lower() not in ["topito"]:  # LINAGORA labeling errors
                assert spkname not in speaker_name, f"Speaker name {spkname} already exists"
            speaker_name.append(spkname)
            speaker_scope.append(spk["@scope"].lower()) if "@scope" in spk else speaker_scope.append("unknown")
            # Fix LINAGORA dataset
            firstname = speaker_name[-1].split("_")[0]
            if firstname in ["julie", "sonia", "celine", "nourine", "sarah"]:
                speaker_gender[-1] = "f"

        speaker_gender = list(map(lambda s: re.sub("^m.*", "m", s), speaker_gender))
        speaker_gender = list(map(lambda s: re.sub("^f.*", "f", s), speaker_gender))

    sections = dict["Trans"]["Episode"]["Section"]
    if "@startTime" in dict["Trans"]["Episode"]["Section"] or "@type" in dict["Trans"]["Episode"]["Section"] or "@endTime" in dict["Trans"]["Episode"]["Section"]:
        sections = [dict["Trans"]["Episode"]["Section"]]
    # print("Length sections: ",len(sections))
    for i in range(len(sections)):
        turns = sections[i]["Turn"]
        if "@startTime" in sections[i]["Turn"]:
            turns = [sections[i]["Turn"]]

        section_topic = sections[i]["@topic"] if "@topic" in sections[i] else "None"
        section_topic = "None" if section_topic == "" else section_topic

        # print("Length turns: ",len(turns))
        for j in range(len(turns)):
            syncs = turns[j]["Sync"]
            if "@time" in turns[j]["Sync"]:
                syncs = [turns[j]["Sync"]]
            # print("Length syncs: ",len(turns))

            turn_start = turns[j]["@startTime"] if "@startTime" in turns[j] else ""
            turn_end = turns[j]["@endTime"] if "@endTime" in turns[j] else ""
            if "@speaker" in turns[j]:
                turn_speaker_ids = turns[j]["@speaker"]
            else:
                turn_speaker_ids = "newSpkGen" + str(newSpkId)
                speaker_id.append(turn_speaker_ids)
                speaker_gender.append("m")  # WTF
                speaker_name.append("unknown")

            if turn_speaker_ids in speaker_id:
                idx = speaker_id.index(turn_speaker_ids)
                turn_speaker_gender = speaker_gender[idx]
                if speaker_name[idx] in ["tous_ensemble"]:
                    continue

            if turn_speaker_ids == "":
                nbr_spk = 0
                turn_speaker_ids = ["-1"]
                continue

            turn_speaker_ids = split_given_list(speaker_id, turn_speaker_ids)
            nbr_spk = len(turn_speaker_ids)

            turn_fidelity = turns[j]["@fidelity"] if "@fidelity" in turns[j] else ""  # (high|medium|low)

            data = []
            num_overlaps = 0
            for k in range(len(syncs)):
                sync_texts = syncs[k]["#text"] if "#text" in syncs[k] else ""
                sync_stime = syncs[k]["@time"] if "@time" in syncs[k] else ""
                sync_texts = to_str(sync_texts)

                # Hack to correctly recover speaker segmentation (1/2)
                sync_texts = re.sub(r"^{{(.*)}}\n", r"{{\1}}", sync_texts)
                sync_texts = re.sub(r"\n{{(.*)}}\n", r"{{\1}} ", sync_texts)
                sync_texts = re.sub(r"\n{{(.*)}}$", r"{{\1}}", sync_texts)
                # sync_texts = re.sub(r"}}\n", "}}", sync_texts)
                sync_texts = re.sub(HACK_EVENTS, "", sync_texts)
                sync_texts = re.sub(r" +", " ", sync_texts)
                sync_texts = re.sub(r"\n+" + HACK_EMPTY_LINES, "\n", sync_texts)
                sync_texts = re.sub(HACK_EMPTY_LINES, "\n", sync_texts)

                if len(sync_texts.strip()) == 0:
                    sync_texts = nbr_spk * [""]
                else:
                    # Hack to correctly recover speaker segmentation (2/2)
                    def split_text(text):
                        if HACK_WHO in sync_texts:
                            texts = re.split(HACK_WHO, text)
                            if text.startswith(HACK_WHO):
                                texts = texts[1:]
                            yield texts
                            yield [s for s in texts if s != ""]
                            yield [s for s in texts if re.sub(r"{{[^}]*}}", "", s).strip() != ""]
                            return

                        for split_pattern in "\n{2,}", "\n{1,}":
                            texts = re.split(split_pattern, text)
                            yield texts
                            yield [s for s in texts if s != ""]
                            yield [s for s in texts if re.sub(r"{{[^}]*}}", "", s).strip() != ""]
                            yield [text]
                        if text != text.strip():
                            for s in split_text(text.strip()):
                                yield s

                    found = False
                    possible_lens = []
                    for sync_texts_ in split_text(sync_texts):
                        if len(sync_texts_) == nbr_spk:
                            found = True
                            break
                        elif len(sync_texts_) not in possible_lens:
                            possible_lens.append(len(sync_texts_))
                    if not found:
                        if 0 in possible_lens and verbose:
                            print("WARNING: skipping empty text with incoherent number of speakers")
                            continue
                        POSSIBLE_LEN_FOR_TWO = [4, 6, 8]
                        if nbr_spk == 2 and max([p in possible_lens for p in POSSIBLE_LEN_FOR_TWO]):
                            for sync_texts_ in split_text(sync_texts):
                                if len(sync_texts_) in POSSIBLE_LEN_FOR_TWO:
                                    text1 = " ".join(sync_texts_[::2])
                                    text2 = " ".join(sync_texts_[1::2])
                                    sync_texts_ = [text1, text2]
                                    break
                        else:
                            if verbose:
                                print(f"WARNING: Inconsistent number of speakers ({nbr_spk}) and texts ({possible_lens}) {turn_start}->{turn_end} {trs_file}")  #:\n{sync_texts}")
                            for sync_texts_ in split_text(sync_texts):
                                break
                            if nbr_spk == 1:
                                sync_texts_[0] = " ".join(sync_texts_)

                            # print("Number of speakers (%d: %s) does not match number of texts (%d: %s) -> %s -> %s" % (nbr_spk, to_str(' '.join(turn_speaker_ids)), len(sync_texts_), to_str(syncs[k]["#text"]), sync_texts, sync_texts_))
                            # import pdb; pdb.set_trace()
                            # raise RuntimeError("Number of speakers (%d: %s) does not match number of texts (%s: %s) -> %s -> %s" % \
                            #                     (nbr_spk, to_str(' '.join(turn_speaker_ids)), possible_lens, to_str(syncs[k]["#text"]), sync_texts, str(list(split_text(sync_texts)))))

                    sync_texts = sync_texts_

                if len(data):
                    # Set end time of previous segments
                    iback = 1
                    while iback <= len(data) and data[-iback]["eTime"] is None:
                        data[-iback]["eTime"] = to_str(sync_stime)
                        iback += 1

                for l, (sync_text, turn_speaker_id) in enumerate(zip(sync_texts, turn_speaker_ids)):
                    if l > 0:
                        num_overlaps += 1

                    idx = speaker_id.index(turn_speaker_id)
                    turn_speaker_gender = speaker_gender[idx]
                    spk_name = speaker_name[idx]

                    if anonymization_level >= 2:
                        spk_name = "spk"
                        spk_index = speaker_index(turn_speaker_id)
                        spkr_id = str(basename) + "_%s-%03d" % (spk_name, spk_index)
                        seg_id = spkr_id
                    else:
                        spkr_id = spk_name
                        if anonymization_level:
                            spkr_id = encrypt_speaker(spkr_id)
                        seg_id = spkr_id + "_" + str(basename)

                    seg_id = "%s_Section%02d_Topic-%s_Turn-%03d_seg-%07d" % (
                        seg_id,
                        i + 1,
                        str(section_topic),
                        j + 1,
                        k + num_overlaps,
                    )

                    if replacements:
                        for pattern, replacement in replacements.items():
                            sync_text = re.sub(r"\b" + pattern + r"\b", replacement, sync_text)

                    sync_text = correct_text(sync_text, capitalize=capitalize, remove_extra_speech=remove_extra_speech)
                    if len(sync_text) == 0:
                        # print(f"WARNING: skipping empty text for {seg_id} ({turn_start}->{turn_end})")
                        continue
                    # if turn_speaker_id not in speaker_id:
                    #     turn_speaker_gender = "m"

                    current = {
                        "id": to_str(seg_id),
                        "spkId": to_str(spkr_id),
                        "spk": to_str(turn_speaker_id),
                        "gender": to_str(turn_speaker_gender),
                        "text": to_str(sync_text),
                        "nbrSpk": nbr_spk,
                        "sTime": to_str(sync_stime),
                        "eTime": None,
                    }

                    data.append(current)

            if len(data):
                # Set end time of previous segments
                iback = 1
                while iback <= len(data) and data[-iback]["eTime"] is None:
                    data[-iback]["eTime"] = to_str(turn_end)
                    iback += 1

            alldata.append(data)

    alldata = [item for sublist in alldata for item in sublist]
    return alldata


def split_given_list(liste, elt):
    if elt in liste:
        return [elt]
    res = []
    liste = sorted(liste, key=lambda x: len(x), reverse=True)
    while len(elt) > 0:
        found = False
        for e in liste:
            assert len(e)
            if elt.startswith(e) and (len(elt) == len(e) or (len(elt) > len(e) and elt[len(e)] == " ")):
                res.append(e)
                elt = elt[len(e) :].strip()
                found = True
                break
        if not found:
            raise RuntimeError(f"Could not find {elt} in {liste}")
    return res


def file_encoding(filename):
    """Guess the encoding of a file"""
    # Note we could use "file" on linux OS
    try:
        import magic

        blob = open(filename, "rb").read()
        m = magic.Magic(mime_encoding=True)
        encoding = m.from_buffer(blob)
        if encoding in ["unknown-8bit"]:
            return xml_encoding(filename)
        return encoding
    except ImportError:
        # print("Warning: magic library not found. Using chardet instead.")
        import chardet

        with open(filename, "rb") as f:
            blob = f.read()
        result = chardet.detect(blob)
        encoding = result["encoding"]
        if encoding in ["unknown-8bit"]:
            return xml_encoding(filename)  # You need to define xml_encoding function
        return encoding


def xml_encoding(infile):
    with open(infile, "rb") as f:
        for header in f:
            break
    header = header.decode("utf-8")
    res = "unknown"
    if 'encoding="' in header:
        res = header.split('encoding="')[1].split('"')[0]
    elif "encoding='" in header:
        res = header.split("encoding='")[1].split("'")[0]
    elif "'-//W3C//DTD XHTML 1.0 Strict//EN'" in header:
        res = "iso-8859-1"
    else:
        res = "utf-8"
    return res


def to_str(s):
    # deprecated
    # if isinstance(s, unicode):
    #     return s.encode('utf-8')
    if isinstance(s, bytes):
        return s.decode("utf-8")
    assert isinstance(s, str)
    return s


def format_speaker_name(spk_name, strong=True):
    # Fix LINAGORA
    if spk_name.lower().startswith("locuteur non ident"):
        spk_name = "unknown"
    if (spk_name.lower().startswith("patrick paysant (") or spk_name.lower().startswith("topito (")) and spk_name.lower().endswith(")"):
        spk_name = spk_name.split("(")[-1].rstrip(")")
    if strong:
        spk_name = spk_name.lower()
    if "(" in spk_name and spk_name.endswith(")") and not spk_name.startswith("("):
        spk_name = spk_name.split("(")[0].strip()
    if strong:
        spk_name = transliterate(spk_name).strip().replace(" ", "_").strip("_")
    # Fix LINAGORA typos
    spk_name = spk_name.replace("jean-pierre_lorra", "jean-pierre_lorre")
    return spk_name


def encrypt_speaker(spk_name):
    """deterministic encryption that cannot be reversed"""
    import hashlib
    import random

    random.seed(1234)
    h = spk_name
    for method in hashlib.sha1, hashlib.sha224, hashlib.sha256, hashlib.sha384, hashlib.sha512, hashlib.md5:
        h = method(h.encode("utf8")).hexdigest()
        h = list(h)
        random.shuffle(h)
        h = "".join(h)[:-1]
    return h


def speaker_index(turn_speaker_id):
    onlydigit = re.sub("[a-zA-Z ]", "", turn_speaker_id)
    try:
        return int(onlydigit)
    except ValueError:
        return 0


_corrections_caracteres_speciaux_fr = [
    (re.compile("%s" % x[0], re.IGNORECASE), "%s" % x[1])
    for x in [
        ("â", "â"),
        ("à", "à"),
        ("á", "á"),
        ("ê", "ê"),
        ("é", "é"),
        ("è", "è"),
        ("ô", "ô"),
        ("û", "û"),
        ("ϊ", "ï"),
        ("î", "î"),
        # Confusion iso-8859-1 <-> utf-8
        (
            " ",
            " ",
        ),
        ("\xc3\x83\xc2\x87", "Ç"),  # "Ã\u0087"
        ("\xc3\x83\xc2\x80", "À"),  # "Ã\u0080"
        ("Ã‰", "É"),
        ("Ãˆ", "È"),
        ("ÃŠ", "Ê"),
        ("Ã”", "Ô"),
        ("Ãœ", "Ü"),
        ("Ã�", "Ï"),
        ("Ã§", "ç"),
        ("Ã©", "é"),
        ("Ã¨", "è"),
        ("Ãª", "ê"),
        ("Ã«", "ë"),
        ("Ã´", "ô"),
        ("Ã¼", "ü"),
        ("Ã¹", "ù"),
        ("Ã®", "î"),
        ("Ã¯", "ï"),
        ("Ã¢", "â"),
        (r"Ã ", "à "),  # WTF
        (r"Â ", " "),  # WTF
        ("disaisâ", "disais Ah"),
        # ("ã","à"),
        # ("Ã","à"),
        # ('À','à'),
        # ('É','é'),
        # ('È','è'),
        # ('Â','â'),
        # ('Ê','ê'),
        # ('Ç','ç'),
        # ('Ù','ù'),
        # ('Û','û'),
        # ('Î','î'),
        # ("œ","oe"),
        # ("æ","ae"),
    ]
]


def correct_text(text, capitalize=True, remove_extra_speech=False):
    # 1. Minimal character normalization
    for reg, replacement in _corrections_caracteres_speciaux_fr:
        text = re.sub(reg, replacement, text)

    text = re.sub("«", '"', text)
    text = re.sub("»", '"', text)
    text = re.sub("“", '"', text)
    text = re.sub("”", '"', text)

    # "#" is forbidden in Kaldi... :(
    text = re.sub("#", "dièse ", text)

    # 2. Extra annotations

    # - "[...]": Annotation about proncunciation or special tags
    # - ex:  +[pron=1 virgule 7 pourcent]
    #        [b]
    text = re.sub(r"\+?\[[^\]]*\]", " ", text)
    # - "&...":  Disfluencies
    # - ex: &hum, §heu
    text = re.sub(r"^[&§]", "", text.strip())
    text = re.sub(r"([' ])[&§]", " ", text)
    # - "(...)": Unsaid stuff
    # - ex: spect(acle)
    text = re.sub(r"\([^\)]*\)", "...", text)
    # - "^^...": Unknown words
    # - ex: ^^Oussila
    text = re.sub(r"\^", "", text)
    # - "*..." : Wrong pronunciation
    # - ex:  *Martin
    text = re.sub(r"\*", "", text)

    if remove_extra_speech:
        text = re.sub(r"<ph_.*/>", "", text)
        text = re.sub(r"{.*}", "", text)

    # # 2. Special character removal
    # text = re.sub('/',' ', text)
    # text = re.sub('#+',' ', text)
    # text = re.sub('\*+', ' ', text)
    # text = re.sub(r"²","", text)
    # text = re.sub(r"\+","", text)

    # Finally, remove extra spaces
    text = collapse_whitespace(text)

    # Capitalize first letter (note: capitalize() converts to lowercase all letters)
    if capitalize and len(text) and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


HACK_EMPTY_LINES = "__HACKEMPTYLINES__"
HACK_EVENTS = "{{HACKEVENTS}}"  # Important to keep {{}} here
HACK_WHO = "__HACKWHO__"


def preformatXML(file, remove_extra_speech):
    file = list(map(lambda s: s.strip(), file))
    file = list(map(lambda s: re.sub('<Who nb=".*"/>', HACK_WHO, s), file))
    file = list(map(lambda s: re.sub("<Sync(.*)/>", r"<Sync\1>", s), file))
    file = list(map(lambda s: re.sub("<Sync", "</Sync><Sync", s), file))
    file = list(map(lambda s: re.sub("</Turn>", "</Sync></Turn>", s), file))
    if remove_extra_speech:
        file = list(map(lambda s: re.sub('<Event.*type="([^"]*)".*/>', HACK_EVENTS, s), file))
        file = list(map(lambda s: re.sub('<Comment.*desc="([^"]*)".*/>', HACK_EVENTS, s), file))
        file = list(map(lambda s: re.sub('<Background.*type="([^"]*)".*/>', HACK_EVENTS, s), file))
        file = list(map(lambda s: HACK_EVENTS if s.strip() in ["(inaudible).", "(inaudible)"] else s, file))  # LINAGORA
        file = list(map(lambda s: re.sub(r" *\(inaudible\)[\. ]?", "... ", s), file))  # LINAGORA
    else:
        file = list(map(lambda s: re.sub('<Event.*type="([^"]*)".*/>', r"{{event:\1}}", s), file))
        file = list(map(lambda s: re.sub('<Comment.*desc="([^"]*)".*/>', r"{{comment:\1}}", s), file))
        file = list(map(lambda s: re.sub('<Background.*type="([^"]*)".*/>', r"{{background:\1}}", s), file))
        file = list(map(lambda s: re.sub(r"\(inaudible\)\.?", "... ", s), file))  # LINAGORA
    file = "\n".join(file)
    file = re.sub("<Turn([^<]*)> </Sync>", r"<Turn\1>", file)

    # file = re.sub('<Event *desc="sampa *: *[^"]*" type="pronounce" extent="begin"/>([^<]*)<Event[^>]*type="pronounce" extent="end"/>',r'@@@@@@\1@@@@',file)
    # file = re.sub('<Event desc="([^"]*)" type="pronounce" extent="begin"/>[^<]*<Event[^>]*type="pronounce" extent="end"/>',r'\1',file)
    # file = re.sub('<Event[^>]*type="([^"]*)"[^>]*/>',r'{{event:\1}}',file)
    # file = re.sub('<Comment[^>]*desc="([^"]*)"[^>]*/>',r'{{comment:\1}}',file)
    # file = re.sub('<Background[^>]*type="([^"]*)"[^>]*/>',r'{{background:\1}}',file)

    # file = re.sub('<Turn([^<]*)startTime="([^<]*)"([^<]*)> *<(?!\bSync\b)([^<]*)>',r'<Turn\1startTime="\2"\3><Sync time="\2"><\4>',file)
    file = re.sub(
        '<Turn([^<]*)startTime="([^<"]*)"([^<]*)> *(?!\b<Sync\b)([^<]+)',
        r'<Turn\1startTime="\2"\3><Sync time="\2">\4',
        file,
    )

    return file
