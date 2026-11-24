import importlib
import os
import unittest

from num2words import CONVERTER_CLASSES

from ssak.utils.text import format_text_ar, format_text_latin, format_text_ru, numbers_and_symbols_to_letters
from ssak.utils.text_basic import replace_ligatures
from ssak.utils.text_numbers import normalize_spoken_numbers
from ssak.utils.text_variants import join_acronyms, normalize_spelling, remove_disfluencies
from ssak.utils.wer import compute_wer

from .utils import Test


class TestImports(Test):
    def test_imports(self):
        # Functions that are (also) imported from these modules
        basic = (
            "collapse_whitespace format_special_characters remove_parenthesis remove_punctuations remove_special_characters "
            "remove_special_words split_around_space_and_apostrophe transliterate"
        ).split()
        for module, names in {
            "ssak.utils.text": basic + ["format_text", "format_text_ar", "format_text_latin", "format_text_ru", "numbers_and_symbols_to_letters"],
            "ssak.utils.text_utils": basic + ["numbers_and_symbols_to_letters"],
            "ssak.utils.text_ar": ["normalize_punct", "symbols_to_letters"],
        }.items():
            for name in names:
                self.assertTrue(callable(getattr(importlib.import_module(module), name)), msg=f"{module}.{name}")


class TestFormatTextLatin(Test):
    def test_format_digits(self):
        self.assertEqual(
            format_text_latin("Ma grand-mère version 2.0 ne connaît pas wav2vec. Commande1"),
            "ma grand-mère version deux point zéro ne connait pas wav to vec commande un",
        )
        self.assertEqual(
            format_text_latin("Elle pesait 67,5kg en 1995, et + en 1996."),
            "elle pesait soixante-sept virgule cinq kilogrammes en mille neuf cent quatre-vingt-quinze et plus en mille neuf cent quatre-vingt-seize",
        )
        self.assertEqual(
            format_text_latin("¾ et 3/4 et 3 m² et 3m³ et 3.7cm² et 3,7 cm³"),
            "trois quarts et trois quarts et trois mètres carrés et trois mètres cubes et trois point sept centimètres carrés et trois virgule sept centimètres cubes",
        )
        self.assertEqual(
            format_text_latin("10 000 et 100 000 000 ne sont pas pareils que 06 12 34 56 78"),
            "dix mille et cent millions ne sont pas pareils que zéro six douze trente-quatre cinquante-six soixante-dix-huit",
        )

        self.assertEqual(
            format_text_latin("10,000 et 100,000,000 et 1111,000,000 et 1111,000000"),
            "dix mille et cent millions et mille cent onze virgule zéro zéro zéro virgule zéro zéro zéro et mille cent onze virgule zéro zéro zéro zéro zéro zéro",
        )

        self.assertEqual(
            format_text_latin("La Rochelle - CAUE 17 2003 2005 Inventaire du Pays"),
            "la rochelle caue dix-sept deux mille trois deux mille cinq inventaire du pays",
        )

        self.assertEqual(
            format_text_latin("Elle emploie 147 collaborateurs et dispose de 22 000 m² de surfaces d' entreposage " "situées Casablanca-Oukacha ( 4 000 m² ) , Had Soualem ( 4 100 m² ) et Ain Sebaâ ( 14 000m² )", remove_parenthesis=False),
            "elle emploie cent quarante-sept collaborateurs et dispose de vingt-deux mille mètres carrés de surfaces d' entreposage "
            "situées casablanca-oukacha quatre mille mètres carrés had soualem quatre mille cent mètres carrés et ain sebaâ quatorze mille mètres carrés",
        )

        # The text in parentheses is removed by default
        self.assertEqual(format_text_latin("Had Soualem (4 100 m²) et Ain Sebaâ"), "had soualem et ain sebaâ")

        self.assertEqual(format_text_latin("1 200 et 01 2002"), "mille deux cents et zéro un deux mille deux")

        self.assertEqual(format_text_latin("1 200 et 01 2002"), "mille deux cents et zéro un deux mille deux")

        self.assertEqual(
            format_text_latin("1/6 000 de seconde pour les  quelques 6 000 précieuses"),
            "un six millième de seconde pour les quelques six mille précieuses",
        )

        self.assertEqual(
            format_text_latin("155/70-12 155/70-15 140/80-13"),
            # "cent cinquante-cinq soixante-dixièmes douze mille cent cinquante-cinq soixante-dix quinze mille cent quarante quatre-vingts treize"
            "cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze mille cent quarante quatre-vingts treize",
        )

        self.assertEqual(
            format_text_latin("Jeu 1 : 160/65-315  155/70-12 155/70-15"),
            # "jeu un cent soixante soixante-cinq mille trois cent quinzièmes cent cinquante-cinq soixante-dixièmes douze cent cinquante-cinq soixante-dixièmes quinze"
            "jeu un cent soixante soixante-cinq millions trois cent quinze mille cent cinquante-cinquièmes soixante-dix douze mille cent cinquante-cinq soixante-dix quinze",
        )

        self.assertEqual(
            format_text_latin("98-151/16, 39-1/16"),
            "quatre-vingt-dix-huit mille cent cinquante et un seize trente-neuf un seizième",
        )

        self.assertEqual(
            format_text_latin("Attentats du 21//07: le suspect extradé d'Italie inculpé à Londres. Nous sommes le 01/01/2019 à 10h00 Il fait -1°C."),
            "attentats du vingt et un juillet le suspect extradé d' italie inculpé à londres nous sommes le premier janvier deux mille dix-neuf à dix heures il fait moins un degrés celsius",
        )

        self.assertEqual(
            format_text_latin("31/01/2019 et 2019/01/01"),
            "trente et un janvier deux mille dix-neuf et premier janvier deux mille dix-neuf",
        )

        self.assertEqual(format_text_latin("32/01/2019 31/13 "), "trente-deux zéro un deux mille dix-neuf trente et un treizièmes")

        self.assertEqual(
            format_text_latin("Une quantité c' est un volume  Quand on divise un volume par une surface on obtient une hauteur Par exemple un 1 litre / m² = 1 000 000 mm3/1 000 000 mm² = 1mm"),
            "une quantité c' est un volume quand on divise un volume par une surface on obtient une hauteur par exemple un un litre mètres carrés égale un million millimètres trois un million millimètres carrés égale un millimètres",
        )

        text = "2" * 700
        expected = ("deux " * len(text)).strip()
        self.assertEqual(format_text_latin(text), expected)

        text = (
            "-" + ("9" * 600) + "19001901190219031904190519061907190819091910191119121913191419151916191719181919192019211922192319241925192619271928192919"
            "3019311932193319341935193619371938193919401941194219431944194519461947194819491950195119521953195419551956195719581959196019"
            "6119621963196419651966196719681969197019711972197319741975197619771978197919801981198219831984198519861987198819891990199119921993199419951996199719981999200020012002200320042005200620072008200920102011201220132014201520162017"
        )
        self.assertEqual(
            format_text_latin(text),
            "moins "
            + " ".join(
                [
                    {
                        "0": "zéro",
                        "1": "un",
                        "2": "deux",
                        "3": "trois",
                        "4": "quatre",
                        "5": "cinq",
                        "6": "six",
                        "7": "sept",
                        "8": "huit",
                        "9": "neuf",
                    }[x]
                    for x in text[1:]
                ]
            ),
        )

    def test_format_decimals_en(self):
        self.assertEqual(
            format_text_latin("It was 1.5 percent, then 3.25% and 0.05.", lang="en"),
            "it was one point five percent then three point two five percent and zero point zero five",
        )
        self.assertEqual(
            format_text_latin("1,250 people, 12,345 others, and 1,000 more", lang="en"),
            "one thousand two hundred and fifty people twelve thousand three hundred and forty-five others and one thousand more",
        )

    def test_format_currencies_en(self):
        for text, expected in {
            "$1.5 million": "one point five million dollars",
            "$5bn and $2.5M": "five billion dollars and two point five million dollars",
            "$1 or £1,250": "one dollar or one thousand two hundred and fifty pounds",
            "$4.99": "four dollars ninety-nine",
        }.items():
            self.assertEqual(format_text_latin(text, lang="en"), expected)

    def test_number_formats(self):
        for text, expected in {
            "3,14159 et 0,0125 et 1.234,56 et 1,234.56": "trois virgule un quatre un cinq neuf et zéro virgule zéro cent vingt-cinq "
            "et mille deux cent trente-quatre virgule cinquante-six et mille deux cent trente-quatre virgule cinquante-six",
            "COVID-19, B-6, le n°5 et il fait -5": "covid dix-neuf b six le numéro cinq et il fait moins cinq",
            "à 14:30, à 7h, à 15h00 et à 1h": "à quatorze heures trente à sept heures à quinze heures et à une heure",
            "la 1ère, la 2nde et les 3èmes": "la première la seconde et les troisièmes",
            "$5, 12,5 € et 1,5 M€": "cinq dollars douze virgule cinq euros et un virgule cinq millions d' euros",
            "X-Prep et XIXe siècle": "x-prep et dix-neuvième siècle",
            "et aujourd'hui 1 injection": "et aujourd'hui un injection",
        }.items():
            self.assertEqual(format_text_latin(text), expected)
        for text, expected in {
            "COVID-19 in the mid-40s and the 1990s": "covid nineteen in the mid forties and the nineteen nineties",
            "at 7:30 or 9:05, 24/7": "at seven thirty or nine oh five twenty-four seven",
            "#5 and No. 6": "number five and number six",
            "1.234,56 and version 1.2.3": "one thousand two hundred and thirty-four point five six and version one dot two dot three",
        }.items():
            self.assertEqual(format_text_latin(text, lang="en"), expected)

    def test_number_spaces(self):
        for lang, texts in {
            "en": {
                "$1.5 millions": "one point five millions dollars",
                "2024": "two thousand and twenty-four",
                "20 24": "twenty twenty-four",
                "in 2021, (12) and 50%.": "in two thousand and twenty-one, (twelve) and fifty percent.",
                "Q3 of US$5bn": "Q three of US five billion dollars",
                "480- uh, 64- 64 to 81": "four hundred and eighty uh, sixty-four sixty-four to eighty-one",
                "5kg  and  6": "five kg  and  six",  # The other spaces are kept
                " 7 ": " seven ",
            },
            "fr": {
                "1, 2, 3 !": "Un, deux, trois !",
                "Il y en a 1,5 % et 20€.": "Il y en a un virgule cinq pour cent et vingt euros.",
                "d'1 dose": "d'un dose",
                "007 à -5°C": "zéro zéro sept à moins cinq degrés Celsius",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(numbers_and_symbols_to_letters(text, lang=lang), expected)

    def test_ordinals(self):
        for lang, texts in {
            "fr": {"les 3es": "les troisièmes"},
            "es": {"el 2º": "el segundo", "la 2ª": "la segunda", "el 3.º": "el tercero"},
            "it": {"il 1º": "il primo", "la 1ª": "la prima"},
            "pt": {"o 2º": "o segundo", "a 2ª": "a segunda"},
            "nl": {"de 2e": "de tweede", "de 1ste": "de eerste"},
            "de": {
                "am 2. Mai": "am zweiten mai",
                "der 3. Oktober": "der dritte oktober",
                "2. Mai": "zweiter mai",
                "1960er Jahre": "neunzehnhundertsechziger jahre",
                "Das sind 2. Und": "das sind zwei und",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected)

    def test_thousands_spaces(self):
        for space in " ", "\u00a0", "\u2009", "\u202f":
            self.assertEqual(numbers_and_symbols_to_letters(f"20{space}000", lang="fr"), "vingt mille")

    def test_titles(self):
        for lang, texts in {
            "fr": {
                "M. Dupont et MM. Durand": "monsieur dupont et messieurs durand",
                "Dr Martin et Dr. Martin": "docteur martin et docteur martin",
                "Pr Durand, Mme Dupont et Mlle. Dupont": "professeur durand madame dupont et mademoiselle dupont",
                "le plan M. est": "le plan m est",
            },
            "en": {
                "Mr. Smith and Dr. Smith": "mister smith and doctor smith",
                "Mrs. Smith, Ms Smith, Prof. Smith": "missus smith miss smith professor smith",
                "Spain vs. Italy": "spain versus italy",
                "no. 5 and No. 6": "number five and number six",
                "he said no. Then": "he said no then",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected)
        self.assertEqual(format_text_latin("Dr. Smith", lang="en", lower_case=False), "Doctor Smith")

    def test_units(self):
        for lang, texts in {
            "fr": {
                "10 km": "dix kilomètres",
                "1 km et 1,5 km": "un kilomètre et un virgule cinq kilomètre",
                "5 km/h et 10m/s": "cinq kilomètres par heure et dix mètres par seconde",
                "10 m, 3 m² et 20 min": "dix mètres trois mètres carrés et vingt minutes",
                "2 m'a": "deux m' a",
            },
            "en": {"10 km": "ten kilometers", "1 km": "one kilometer", "60 mph and 5 kg": "sixty miles per hour and five kilograms"},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected)

    def test_acronyms(self):
        # The case is kept, unless lower case is asked
        for text, expected in {
            "the U.S. and the U.K. are": "the US and the UK are",
            "in the U.S.": "in the US.",
            "la S.N.C.F. et l'O.N.U, oui": "la SNCF et l'ONU, oui",
            "J. Smith. A. B and end. I am.": "J. Smith. A. B and end. I am.",
            "www.a.b.com v1.2": "www.a.b.com v1.2",
        }.items():
            self.assertEqual(join_acronyms(text), expected)
        self.assertEqual(format_text_latin("la S.N.C.F. est"), "la sncf est")
        self.assertEqual(format_text_latin("la S.N.C.F. est", lower_case=False), "la SNCF est")
        self.assertEqual(format_text_latin("the U.S. is", lang="en"), "the us is")
        self.assertEqual(format_text_latin("the U.S. is", lang="en", lower_case=False), "the US is")
        self.assertEqual(format_text_latin("the U.S. is big.", lang="en", lower_case=False, keep_punc=True), "the US is big.")

    def test_spelling_variants(self):
        for lang, texts in {
            "fr": {
                "L'événement, les événements": "l' évènement les évènements",
                "le maître, la maîtresse et les maîtres": "le maitre la maitresse et les maitres",
                "il connaît, il paraît, s'il vous plaît, naître": "il connait il parait s' il vous plait naitre",
                "la boîte, les chaînes et l'île": "la boite les chaines et l' ile",
                "goût, coûter, août, brûlé et piqûre": "gout couter aout brulé et piqure",
                "ambiguë, ambigüe et ambiguïté": "ambigue ambigue et ambiguité",
                "OK pour le week-end": "okay pour le weekend",
                # Not changed
                "dû, mûr, sûr, jeûne, il fût, nous fîmes, Nîmes, Benoît": "dû mûr sûr jeûne il fût nous fîmes nîmes benoît",
            },
            "en": {
                "the colours of my neighbour's coloured favourite": "the colors of my neighbor' s colored favorite",
                "organise, organised, the organisation, realising": "organize organized the organization realizing",
                "analyse, travelled, cancelling, modelling": "analyze traveled canceling modeling",
                "the centre, 5 metres, a grey theatre": "the center five meters a gray theater",
                "the defence programme, OK, e-mail": "the defense program okay email",
                "paediatric and anaemia": "pediatric and anemia",
                # Not changed
                "our hour, four of your, optimism, optimist, promise, exercise, emphasis, wise, analyses": "our hour four of your optimism optimist promise exercise emphasis wise analyses",
                "glamour, contour, tour, devour, programmed, controlled, called, sell": "glamour contour tour devour programmed controlled called sell",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected)
        self.assertEqual(normalize_spelling("Maître MAÎTRE maître Île", "fr"), "Maitre MAITRE maitre Ile")
        self.assertEqual(normalize_spelling("Colour COLOUR Organise ORGANISE Centre", "en"), "Color COLOR Organize ORGANIZE Center")
        self.assertEqual(normalize_spelling("événement", "es"), "événement")

    def test_disfluencies(self):
        for lang, texts in {
            "fr": {
                "Euh, le matin": "Le matin",
                "le matin, euh, et le soir": "le matin, et le soir",
                "le euh matin hum hmm mmh": "le matin",
                "Le matin, euh.": "Le matin.",
                "Euh. Oui euuuh heu non": "Oui non",
                "un humain, l'euh, heureux, Meuh, euh-là": "un humain, l'euh, heureux, Meuh, euh-là",
                "le um uh": "le um uh",
            },
            "en": {
                "Uh, yes": "Yes",
                "So um we, uh, go hmm": "So we, go",
                "Um... I, erm, see.": "I, see.",
                "uhh umm mmm ok": "ok",
                # Not removed
                "uh-huh, mhm, oh, ah, er, hum a tune, umbrella": "uh-huh, mhm, oh, ah, er, hum a tune, umbrella",
            },
            "de": {"ähm um zwei Uhr": "um zwei Uhr"},
            "pt": {"hmm um dia": "um dia"},
            "ru": {"euh": "euh"},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(remove_disfluencies(text, lang), expected)
        # Not by default in format_text
        self.assertEqual(format_text_latin("euh le hum matin"), "euh le hum matin")
        self.assertEqual(format_text_latin("euh le hum matin", remove_disfluencies=True), "le matin")
        # By default in the WER
        for ref, pred, normalization in [("euh le matin", "le matin hum", "fr"), ("uh yes", "yes um", "en+")]:
            self.assertEqual(compute_wer([ref], [pred], normalization=normalization)["wer"], 0)
            self.assertGreater(compute_wer([ref], [pred], normalization=normalization, remove_disfluencies=False)["wer"], 0)
        self.assertGreater(compute_wer(["uh yes"], ["yes"])["wer"], 0)  # no normalization

    def test_wer_variants(self):
        for lang, texts in {
            "fr": [
                ("M. Dupont", "monsieur Dupont"),
                ("10 km", "dix kilomètres"),
                ("5 km/h", "cinq kilomètres par heure"),
                ("la S.N.C.F.", "la SNCF"),
                ("l'événement", "l'évènement"),
                ("le maître", "le maitre"),
                ("les 3es", "les troisièmes"),
            ],
            "en": [
                ("October 7", "October seventh"),
                ("October 7", "October the seventh"),
                ("May 21", "May twenty-first"),
                ("in 1789", "in seventeen eighty-nine"),
                ("in 1066", "in ten sixty-six"),
                ("in 2030", "in twenty thirty"),
                ("in 1905", "in nineteen oh-five"),
                ("in 1900", "in nineteen hundred"),
                ("the U.S.", "the US"),
                ("Mr. Smith", "mister Smith"),
                ("10 km", "ten kilometers"),
                ("colour", "color"),
                ("organise", "organize"),
            ],
            "es": [
                ("200 casas", "doscientas casas"),
                ("500 casas", "quinientas casas"),
                ("21 personas", "veintiuna personas"),
                ("2º", "segundo"),
            ],
            "it": [("1º", "primo"), ("1 casa", "una casa")],
            "pt": [("200 casas", "duzentas casas"), ("2 casas", "duas casas"), ("16", "dezesseis"), ("16", "dezasseis"), ("2º", "segundo")],
            "de": [
                ("1 Haus", "ein Haus"),
                ("1998", "neunzehnhundertachtundneunzig"),
                ("1998", "eintausendneunhundertachtundneunzig"),
                ("am 2. Mai", "am zweiten Mai"),
                ("1960er Jahre", "neunzehnhundertsechziger Jahre"),
            ],
            "nl": [
                ("1998", "negentienachtennegentig"),
                ("1998", "negentienhonderd achtennegentig"),
                ("1998", "duizendnegenhonderdachtennegentig"),
                ("de 2e", "de tweede"),
            ],
        }.items():
            for ref, pred in texts:
                for normalization in [lang, lang + "+"]:
                    self.assertEqual(compute_wer([ref], [pred], normalization=normalization)["wer"], 0, msg=f"{ref} / {pred} ({normalization})")
        # Numbers that are not years, not changed
        self.assertEqual(normalize_spoken_numbers("one hundred twenty twenty one", "en"), "one hundred twenty twenty one")
        self.assertEqual(normalize_spoken_numbers("twenty twenty thousand", "en"), "twenty twenty thousand")
        # What was said is different
        for ref, pred in [("don't", "do not"), ("gonna", "going to"), ("in 1789", "in 1798")]:
            self.assertGreater(compute_wer([ref], [pred], normalization="en+")["wer"], 0)

    def test_roman_numbers(self):
        for lang, texts in {
            "fr": {
                "la Ve République": "la cinquième République",
                "le VIIIe siècle, le XXe": "le huitième siècle, le vingtième",
                "les XVIIIes": "les dix-huitièmes",
                "François Ier": "François premier",
                "Ire partie, Ière partie": "première partie, première partie",
                "Louis XIV et Henri IV": "Louis quatorze et Henri quatre",
                "le chapitre II": "le chapitre deux",
                # Not converted
                "Vers le Ver, Vie, Ie, Vre, Xer": "Vers le Ver, Vie, Ie, Vre, Xer",
                "Vitamine C, X-Prep, V, I": "Vitamine C, X-Prep, V, I",
                "en IV, injection IV, CI et II": "en IV, injection IV, CI et II",
            },
            "en": {
                "World War II": "World War two",
                "Rocky II": "Rocky two",
                "Henry VIII": "Henry the eighth",
                "the XIXth century": "the nineteenth century",
                # Not converted
                "World War I think, Charles I": "World War I think, Charles I",
                "IV fluids, Give IV, Ist": "IV fluids, Give IV, Ist",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(numbers_and_symbols_to_letters(text, lang=lang), expected)
        for lang, ref, pred in [("fr", "la Ve République", "la cinquième République"), ("fr", "François Ier", "François premier"), ("en", "Henry VIII", "Henry the eighth")]:
            self.assertEqual(compute_wer([ref], [pred], normalization=lang + "+")["wer"], 0)

    def test_degrees_or_ordinals(self):
        for lang, texts in {
            "es": {
                "el 2° piso": "el segundo piso",
                "es el 5°": "es el quinto",
                "quemadura de 2° grado": "quemadura de segundo grado",
                "1° de mayo": "primero de mayo",
                "la 2° vez": "la segunda vez",
                "Carlos, 2° clasificado": "Carlos, segundo clasificado",
                "hace 10°": "hace diez grados",
                "a 5° esta mañana": "a cinco grados esta mañana",
                "un ángulo de 10° respecto al eje": "un ángulo de diez grados respecto al eje",
                "entre 5° y 10°": "entre cinco grados y diez grados",
                "a 10° bajo cero": "a diez grados bajo cero",
                "10° de latitud": "diez grados de latitud",
                "20 °C": "veinte grados Celsius",
                "2 °C": "dos grados Celsius",
                "a 45°": "a cuarenta y cinco grados",
                "el 10º": "el décimo",
            },
            "it": {
                "il 2° piano": "il secondo piano",
                "1° maggio": "primo maggio",
                "è il 5°": "è il quinto",
                "fa 10°": "fa dieci gradi",
                "ci sono 5° stamattina": "ci sono cinque gradi stamattina",
                "10°": "dieci gradi",
                "10° sotto zero": "dieci gradi sotto zero",
                "fa 10° oggi": "fa dieci gradi oggi",
            },
            "pt": {
                "o 2° andar": "o segundo andar",
                "1° de maio": "primeiro de maio",
                "faz 10°": "faz dez graus",
                "subiu 2° durante a noite": "subiu dois graus durante a noite",
                "ficou em 2° lugar": "ficou em segundo lugar",
                "10° abaixo de zero": "dez graus abaixo de zero",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(numbers_and_symbols_to_letters(text, lang=lang), expected)

    def test_numbers_at_sentence_start(self):
        for lang, texts in {
            "fr": {
                "3 comprimés.": "Trois comprimés.",
                "Il dit oui. 3 fois": "Il dit oui. Trois fois",
                "5 € par jour.": "Cinq euros par jour.",
                # No case nor punctuation, or not the start of a sentence
                "3 comprimés": "trois comprimés",
                "Il dit 3 fois": "Il dit trois fois",
                "Il dit oui... 3 fois": "Il dit oui... trois fois",
                "Cf. 3 fois": "Cf. trois fois",
            },
            "en": {"$5 a day.": "Five dollars a day.", "No. 5 is here.": "Number five is here.", "He is No. 5.": "He is number five."},
            "de": {"3 Häuser.": "Drei Häuser."},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(numbers_and_symbols_to_letters(text, lang=lang), expected)
        self.assertEqual(format_text_latin("3 comprimés."), "trois comprimés")
        self.assertEqual(format_text_latin("3 comprimés.", lower_case=False), "Trois comprimés")
        self.assertEqual(format_text_latin("3 comprimés.", lower_case=False, keep_punc=True), "Trois comprimés.")

    def test_spelling_variants_pt(self):
        for text, expected in {
            "econômico Antônio gênero": "económico António género",
            "contato, recepção, úmido": "contacto, receção, húmido",
            "Dezesseis registros": "Dezasseis registos",
            # Not changed: same in Portugal, or other words
            "fêmea, estômago, você, têm": "fêmea, estômago, você, têm",
            "caminhão, gol, equipe, planejar": "caminhão, gol, equipe, planejar",
        }.items():
            self.assertEqual(normalize_spelling(text, "pt"), expected)
        self.assertEqual(format_text_latin("16 gêneros", lang="pt"), "dezasseis géneros")

    def test_wer_dates_years_decimals(self):
        for lang, ref, pred in [
            ("en", "7 October", "seventh October"),
            ("en", "7 October", "the seventh of October"),
            ("en", "the 7th of October", "the seventh of October"),
            ("en", "21 May", "twenty-first May"),
            ("en", "the 2000s", "the two thousands"),
            ("en", "3.25", "three point twenty-five"),
            ("en", "3.10", "three point ten"),
            ("en", "0.05", "zero point oh five"),
            ("fr", "0,025", "zéro virgule zéro deux cinq"),
            ("fr", "0,025", "zéro virgule zéro vingt-cinq"),
            ("fr", "3,125", "trois virgule un deux cinq"),
            ("nl", "in 1905", "in negentienhonderdvijf"),
            ("nl", "in 1900", "in negentienhonderd"),
            ("pt", "16", "dezesseis"),
            ("pt", "16", "dezasseis"),
        ]:
            self.assertEqual(compute_wer([ref], [pred], normalization=lang + "+")["wer"], 0, msg=f"{ref} / {pred}")
        for lang, ref, pred in [("fr", "0,025", "zéro virgule deux cinq"), ("en", "7 October", "eighth October"), ("nl", "1905", "negentienhonderdzes")]:
            self.assertGreater(compute_wer([ref], [pred], normalization=lang + "+")["wer"], 0, msg=f"{ref} / {pred}")

    def test_wer_parentheses(self):
        # The words in parentheses are kept
        for lang, ref, pred, wrong in [
            ("fr", "le matin (et le soir) oui", "le matin et le soir oui", "le matin oui"),
            ("en", "the morning (and evening) yes", "the morning and evening yes", "the morning yes"),
            ("ru", "утром (и вечером) да", "утром и вечером да", "утром да"),
        ]:
            for normalization in [lang, lang + "+"]:
                self.assertEqual(compute_wer([ref], [pred], normalization=normalization)["wer"], 0, msg=f"{ref} ({normalization})")
                self.assertGreater(compute_wer([ref], [wrong], normalization=normalization)["wer"], 0, msg=f"{ref} ({normalization})")

    def test_wer_arabic_case(self):
        # The case of the Latin words does not count in the WER
        self.assertEqual(compute_wer(["يقول Hello"], ["يقول hello"], normalization="ar")["wer"], 0)

    def test_ligatures(self):
        self.assertEqual(format_text_latin("Œuvre, cœur, ex æquo, aﬀaire, eﬃcace, ﬁn, reﬂet"), "oeuvre coeur ex aequo affaire efficace fin reflet")
        self.assertEqual(replace_ligatures("Œuvre ŒUVRE Œ"), "Oeuvre OEUVRE Oe")

    def test_diacritics_other_languages(self):
        for lang, text in {"es": "el año pasado qué pingüino", "de": "größe für öl", "it": "perché è così", "pt": "não é você coração", "nl": "één café"}.items():
            self.assertEqual(format_text_latin(text, lang=lang), text)

    def test_number_other_languages(self):
        for lang, expected in {
            "es": "mil quinientos euros y cincuenta por ciento",
            "de": "eintausendfünfhundert euro y fünfzig prozent",
            "it": "millecinquecento euro y cinquanta per cento",
            "pt": "mil e quinhentos euros y cinquenta por cento",
            "nl": "duizendvijfhonderd euro y vijftig procent",
        }.items():
            self.assertEqual(format_text_latin("1.500 € y 50%", lang=lang), expected)
        # Unsupported language: not converted
        self.assertEqual(numbers_and_symbols_to_letters("21 1,5 50%", lang="xx"), "21 1,5 50%")

    def test_wer_numbers(self):
        for lang, texts in {
            "fr": [
                ("il prend 1,5 mg", "il prend un virgule cinq milligrammes"),
                ("il prend 1,5 mg", "il prend 1.5 mg"),
                ("1 goutte", "une goutte"),
                ("50 pourcent", "50 %"),
                ("quatre-vingts et 200", "quatre-vingt et deux cent"),
                ("en 1980", "en dix-neuf cent quatre-vingt"),
                ("à 14:30", "à quatorze heures trente"),
            ],
            "en": [
                ("it was 250 people", "it was two hundred fifty people"),
                ("in 2021", "in twenty twenty-one"),
                ("in 1905 and 2005", "in nineteen oh five and two thousand and five"),
                ("$1.5 million", "one point five million dollars"),
                ("3.25%", "three point two five per cent"),
                ("1,200 and a hundred", "twelve hundred and one hundred"),
            ],
        }.items():
            for ref, pred in texts:
                for normalization in [lang, lang + "+"]:
                    self.assertEqual(compute_wer([ref], [pred], normalization=normalization)["wer"], 0, msg=f"{ref} / {pred} ({normalization})")
            self.assertGreater(compute_wer(["1,5 et 99"], ["15 et 99"], normalization=lang)["wer"], 0)
        # The inputs are not modified
        refs, preds = ["a euh b"], ["a b"]
        compute_wer(refs, preds, replacements_ref={"euh": ""}, replacements_pred={"euh": ""})
        self.assertEqual(refs, ["a euh b"])

    def test_normalize_english_numbers(self):
        for text, expected in {
            "two hundred and fifty and rock and roll": "two hundred fifty and rock and roll",
            "a million and a half": "one million and a half",
            "in twenty twenty-one and nineteen ninety five": "in two thousand twenty one and one thousand nine hundred ninety five",
            "in twenty twenty": "in two thousand twenty",
            "in two thousand and twenty-one": "in two thousand twenty one",
        }.items():
            self.assertEqual(normalize_spoken_numbers(text, "en"), expected)

    def test_format_currencies(self):
        for text in [
            "ça coute 1,20€",
            "ça coute 1,20 €",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un euros vingt",  # TODO: remove the s...
            )

        for text in [
            "ça coute 1,20$",
            "ça coute 1,20 $",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un dollars vingt",  # TODO: remove the s...
            )

        for text in [
            "ça coute 1.20$ * 6 mois",
            "ça coute 1.20 $ * 6 mois",
        ]:
            self.assertEqual(format_text_latin(text), "ça coute un point vingt dollars six mois")

    def test_format_special_chars(self):
        self.assertEqual(format_text_latin("L’état “from scratch”."), "l' état from scratch")

        self.assertEqual(
            format_text_latin(
                "``Elle veut nous faire croire qu'elle était complètement abrutie'', a affirmé l'avocat général "
                "notant cependant que l'ancienne prostituée -qui aurait dit dans la voiture ``Vas-y,allume-les!''-avait réussi sa reconversion après 1994."
            ),
            "elle veut nous faire croire qu' elle était complètement abrutie a affirmé l' avocat général "
            "notant cependant que l' ancienne prostituée qui aurait dit dans la voiture vas-y allume-les avait réussi sa reconversion après mille neuf cent quatre-vingt-quatorze",
        )

        self.assertEqual(
            format_text_latin("Hello aaaf-bearn-gascogne, belle-mère, rouge-gorge."),
            "hello aaaf bearn gascogne belle-mère rouge-gorge",
        )

    def test_websites(self):
        self.assertEqual(
            format_text_latin("http://www.linagora.blah.com/page.html est un site / www.linagora.com en est un autre"),
            "http deux points slash slash www point linagora point blah point com slash page point html est un site www point linagora point com en est un autre",
        )

        self.assertEqual(format_text_latin("www.len.com len.. ralenti"), "www point len point com len ralenti")

    def test_abbr(self):
        self.assertEqual(
            format_text_latin("M. Dupont, Mme. Dupont, Mlle. Dupont"),
            "monsieur dupont madame dupont mademoiselle dupont",
        )
        self.assertEqual(
            format_text_latin("C'est la <DATE> est au format aaaa-mm-dd. ça mesure 3mm"),
            "c' est la date est au format aaaa-millimètres-dd ça mesure trois millimètres",  # Not the best...
        )

    def test_spelling(self):
        self.assertEqual(
            format_text_latin("J'ai mis les clefs dans la serrure en revenant du bistrot, etc. Je suis un feignant et cætera."),
            "j' ai mis les clés dans la serrure en revenant du bistro et cetera je suis un fainéant et cetera",
        )

    def test_misc(self):
        self.assertEqual(format_text_latin("chef-d'oeuvre et- voilà - quoi"), "chef-d' oeuvre et voilà quoi")

    def test_disable_some(self):
        self.assertEqual(
            format_text_latin("Œuf 6Go 6ème 2,6.", convert_numbers=False, lower_case=False, keep_punc=True),
            "Oeuf 6 Go 6ème 2,6.",
        )
        self.assertEqual(
            format_text_latin("Œuf 6Go 6ème 2,6.", convert_numbers=False, lower_case=False, keep_punc=False),
            "Oeuf 6 Go 6ème 2 6",
        )

    def test_non_regression_fr(self):
        output_file = self.get_temp_path("output.txt")
        acronym_file = self.get_temp_path("acronyms.txt")
        special_char_file = self.get_temp_path("special_chars.txt")
        for f in [output_file, acronym_file, special_char_file]:
            if os.path.exists(f):
                os.remove(f)
        self.assertRun(
            [
                self.get_tool_path("clean_text_fr.py"),
                self.get_data_path("text/frwac.txt"),
                output_file,
                "--empty_string_policy=ignore",
                "--linebreak_policy=allow",
                "--extract_parenthesis",
                "--wer_format",
                "--file_acro",
                acronym_file,
                "--file_special",
                special_char_file,
            ]
        )
        self.assertNonRegression(output_file, "format_text/output.txt")
        self.assertNonRegression(acronym_file, "format_text/acronyms.txt")
        self.assertNonRegression(special_char_file, "format_text/special_chars.txt")
        for f in [output_file, acronym_file, special_char_file]:
            os.remove(f)


class TestFormatTextArabic(Test):
    def test_options(self):
        sentence = 'في اللغة الإنجليزية ، يمكن للمرء أن يقول "Hello world"!'

        self.assertEqual(format_text_ar(sentence, keep_punc=False, keep_latin_chars=False), "في اللغة الإنجليزية يمكن للمرء أن يقول")

        self.assertEqual(
            format_text_ar(sentence, keep_punc=True, keep_latin_chars=False),
            'في اللغة الإنجليزية ، يمكن للمرء أن يقول " "!',
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=False, keep_latin_chars=True),
            "في اللغة الإنجليزية يمكن للمرء أن يقول Hello world",
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=True, keep_latin_chars=True),
            sentence,  # "في اللغة الإنجليزية ، يمكن للمرء أن يقول \"Hello world\"!"
        )

        self.assertEqual(format_text_ar(sentence, lower_case=True), "في اللغة الإنجليزية يمكن للمرء أن يقول hello world")

        sentence = "؟Jérôme。"

        self.assertEqual(format_text_ar(sentence, keep_punc=False, keep_latin_chars=False), "")
        self.assertEqual(format_text_ar(sentence, keep_punc=True, keep_latin_chars=False), "؟")
        self.assertEqual(
            format_text_ar(sentence, keep_punc=True, keep_latin_chars=True),
            sentence,  # "؟Jérôme。"
        )
        self.assertEqual(format_text_ar(sentence, keep_punc=False, keep_latin_chars=True), "Jérôme")

    def test_format_digits(self):
        self.assertEqual(
            format_text_ar("بعض الأرقام: 01 و 314 و 315.5 و ۰۹ و ۹۰"),
            "بعض الأرقام صفر واحد و ثلاثمائة و أربعة عشر و ثلاثمائة و خمسة عشر فاصيله خمسة و صفر تسعة و تسعون",
        )

        self.assertEqual(
            format_text_ar(
                "يوجد 10000 شخص ، عنوان IP الخاص بي هو 951.357.123 ، ورقم هاتفي هو 06 12 34 56 78",
                keep_latin_chars=False,
            ),
            "يوجد عشرة آلاف شخص عنوان الخاص بي هو تسعمائة و واحد و خمسون فاصيله ثلاثمائة و سبعة و خمسون فاصيله مائة و ثلاثة و عشرون ورقم هاتفي هو صفر ستة اثنا عشر أربعة و ثلاثون ستة و خمسون ثمانية و سبعون",
        )

        self.assertEqual(format_text_ar("وأغلق بسعر $45.06 للبرميل."), "وأغلق بسعر دولار خمسة و أربعون فاصيله صفر ستة للبرميل")

    def test_symbols_converting(self):
        self.assertEqual(
            format_text_ar("للعام 1435/ 1436هـ"),
            "للعام ألف و أربعمائة و خمسة و ثلاثون ألف و أربعمائة و ستة و ثلاثون هجري",
        )

        self.assertEqual(format_text_ar("7 ق.م"), "سبعة قبل الميلاد")

        self.assertEqual(
            format_text_ar("300¥ = 300$ = 300£ = 300₹"),
            "ثلاثمائة ين يساوي ثلاثمائة دولار يساوي ثلاثمائة جنيه يساوي ثلاثمائة روبية هندية",
        )

    def test_dates(self):
        self.assertEqual(format_text_ar("هجري 1437/01/20"), "هجري ألف و أربعمائة و سبعة و ثلاثون محرم عشرون")

        self.assertEqual(format_text_ar("هجري 1937/01/20"), "هجري عشرون يناير ألف و تسعمائة و سبعة و ثلاثون")

        self.assertEqual(format_text_ar("اليوم 22/04/2002"), "اليوم اثنان و عشرون أبريل ألفان و اثنان")

    def test_biggest_numbers(self):
        self.assertEqual(
            format_text_ar("-123456789123456789128942"),
            "سالب مائة و ثلاثة و عشرون سكستيليونا و أربعمائة و ستة و خمسون كوينتليونا و سبعمائة و تسعة و ثمانون كوادريليونا و مائة و ثلاثة و عشرون تريليونا و"
            " أربعمائة و ستة و خمسون مليارا و سبعمائة و تسعة و ثمانون مليونا و مائة و ثمانية و عشرون ألفا و تسعمائة و اثنان و أربعون",
        )

        self.assertEqual(
            format_text_ar("-1234567891234567891289425"),
            "سالب سبتيليون و مئتان و أربعة و ثلاثون سكستيليونا و خمسمائة و سبعة و ستون كوينتليونا و ثمانمائة و واحد و تسعون كوادريليونا و مئتان و أربعة و"
            " ثلاثون تريليونا و خمسمائة و سبعة و ستون مليارا و ثمانمائة و واحد و تسعون مليونا و مئتان و تسعة و ثمانون ألفا و أربعمائة و خمسة و عشرون",
        )

    @unittest.skipUnless("ar_tn" in CONVERTER_CLASSES, "this version of num2words does not support Tunisian Arabic")
    def test_tunisian_numbers(self):
        self.assertEqual(format_text_ar("اليوم 22/04/2002", lang="ar_tn"), "اليوم ثنين و عشرين أفريل ألفين و ثنين")

        self.assertEqual(
            format_text_ar("-123456789123456789128942", lang="ar_tn"),
            "ناقس مياة و ثلاث و عشرين سكستيليون و أربعمياة و ستة و خمسين كوينتليون و سبعمياة و تسع و ثمانين كوادريليون و مياة و ثلاث و"
            " عشرين تريليون و أربعمياة و ستة و خمسين مليار و سبعمياة و تسع و ثمانين مليون و مياة و ثمني و عشرين ألف و تسعمياة و ثنين و أربعين",
        )

        self.assertEqual(format_text_ar("-12", lang="ar_tn"), "ناقس أثناش")

        self.assertEqual(format_text_ar("-5000", lang="ar_tn"), "ناقس خمس آلاف")

    def test_words_normalization(self):
        sentence = "انشا الله  بسبعه عسلامة يسلمك ديجا"

        self.assertEqual(
            format_text_ar(sentence, keep_punc=False, keep_latin_chars=False, lang="ar_tn", normalize_dialect_words=True),
            "إن شاء الله بسبع عالسلامة يسلمك",
        )

        self.assertEqual(
            format_text_ar(sentence, keep_punc=False, keep_latin_chars=True, lang="ar_tn", normalize_dialect_words=True),
            "إن شاء الله بسبع عالسلامة يسلمك déjà",
        )

    def test_chars_normalization(self):
        sentence = "بڨداش أﻛ الخبزﻫ"

        self.assertEqual(format_text_ar(sentence, lang="ar_tn", normalize_dialect_words=True), "بقداش أك الخبزه")

    def test_remove_repeated_ar_chars(self):
        sentence = "صببباح النووووور Meeeerci"

        self.assertEqual(format_text_ar(sentence, keep_latin_chars=True), "صبباح النوور Meeeerci")

    def test_remove_outer_apostrophes_and_hyphens(self):
        sentence = "Here's an example: remove 'these ' apostrophes and - these -hyphens but not don't or high-quality"

        self.assertEqual(
            format_text_ar(sentence, keep_latin_chars=True),
            "Here's an example remove these apostrophes and these hyphens but not don't or high-quality",
        )

    # def test_digit_round_check(self):

    #     from ssak.utils.language import translate_language
    #     from text_to_num import alpha2digit

    #     def _round(num, lang="fr"):
    #         text_ar = format_text_ar(str(num))
    #         text_fr = translate_language(text_ar, dest=lang, src="ar")
    #         num_str = alpha2digit(text_fr, lang=lang)
    #         return int(num_str)

    #     def _test_round(num, lang="fr"):
    #         self.assertEqual(_round(num, lang=lang), num)

    #     _test_round(123)
    #     _test_round(9128942)
    #     _test_round(789128942)
    #     _test_round(6789128942)


class TestFormatTextRu(Test):
    def test_format_digits(self):
        self.assertEqual(
            format_text_ru("2 января и 02 января и 02 января 2001 и 2001 г"),
            "второго января и второго января и второго января две тысячи первого и две тысячи первого года",
        )
        self.assertEqual(
            format_text_ru("Тест №1, Пушкин родился 26 мая (6 июня) 1799 г. в Москве, в Немецкой слободе."),
            "тест номер один пушкин родился двадцать шестого мая шестого июня одна тысяча семьсот девяносто девятого года в москве в немецкой слободе",
        )
        self.assertEqual(
            format_text_ru("29 августа, 11 сентября, 26 мая, 1 апреля, 2 января, 31 декабря"),
            "двадцать девятого августа одиннадцатого сентября двадцать шестого мая первого апреля второго января тридцать первого декабря",
        )
        self.assertEqual(
            format_text_ru("в четверг 4-го числа в 4 часа регулировщик регулировал в Лигурии, но 36%  и 6,5 кг лавировали"),
            "в четверг четвертого числа в четыре часа регулировщик регулировал в лигурии но тридцать шесть процентов и шесть запятая пять килограмм лавировали",
        )
        self.assertEqual(
            format_text_ru("Бутербрд: 1 хлеб, --------------- 40 яблок, 21мл соуса, 8,5 помидоров, 37г соли"),
            "бутербрд один хлеб сорок яблок двадцать один миллилитр соуса восемь запятая пять помидоров тридцать семь г соли",
        )
        self.assertEqual(
            format_text_ru("2378600, 78900532, 67588849214"),
            "два миллиона триста семьдесят восемь тысяч шестьсот семьдесят восемь миллионов девятьсот тысяч пятьсот тридцать два шестьдесят семь миллиардов пятьсот восемьдесят восемь миллионов восемьсот сорок девять тысяч двести четырнадцать",
        )

    def test_options(self):
        sentence = "Я помню чудное мгновенье, передо мной явилась ты. Когда-нибудь Ёжики и ёлки. This is a latin text to test. А это — определение"

        self.assertEqual(
            format_text_ru(
                sentence,
                keep_punc=True,
                lower_case=False,
                force_transliteration=False,
                remove_optional_diacritics=False,
            ),
            "Я помню чудное мгновенье, передо мной явилась ты. Когда-нибудь Ёжики и ёлки. This is a latin text to test. А это — определение",
        )
        self.assertEqual(
            format_text_ru(sentence, keep_punc=False, lower_case=True, force_transliteration=True, remove_optional_diacritics=True),
            "я помню чудное мгновенье передо мной явилась ты когда нибудь ежики и елки this is a latin text to test а это определение",
        )
        self.assertEqual(
            format_text_ru(
                sentence,
                keep_punc=False,
                keep_hyphen=True,
                lower_case=True,
                force_transliteration=True,
                remove_optional_diacritics=True,
            ),
            "я помню чудное мгновенье передо мной явилась ты когда-нибудь ежики и елки this is a latin text to test а это определение",
        )

        sentence = "Now close your eyes, there goes the transliteration"

        self.assertEqual(
            format_text_ru(sentence, keep_punc=False, lower_case=True, force_transliteration=True, remove_optional_diacritics=True),
            "нов клосе ёур еес зере гоес зе транслитератион",
        )

    def test_symbols(self):
        self.assertEqual(
            format_text_ru("10 * 2 = 20. 20% от 500€ = 100€"),
            "десять умножить на два равно двадцать двадцать процентов от пятьсот евро равно сто евро",
        )
        self.assertEqual(format_text_ru("svetlana21@mail.ru"), "светлана двадцать один собака маилру")
        self.assertEqual(
            format_text_ru("M&Ms. 21 м³. 451°F"),
            "m энд ms двадцать один метр в кубе четыреста пятьдесят один градус по фаренгейту",
        )
