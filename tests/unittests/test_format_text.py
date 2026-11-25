import importlib
import os
import unittest
import warnings

from num2words import CONVERTER_CLASSES

from ssak.utils.text import format_text_ar, format_text_latin, format_text_ru, normalize_text, numbers_and_symbols_to_letters
from ssak.utils.text_basic import replace_ligatures
from ssak.utils.text_numbers import normalize_spoken_numbers
from ssak.utils.text_variants import _UNITS, join_acronyms, normalize_spelling, remove_disfluencies
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
            format_text_latin("Elle emploie 147 collaborateurs et dispose de 22 000 m² de surfaces d' entreposage " "situées Casablanca-Oukacha ( 4 000 m² ) , Had Soualem ( 4 100 m² ) et Ain Sebaâ ( 14 000m² )"),
            "elle emploie cent quarante-sept collaborateurs et dispose de vingt-deux mille mètres carrés de surfaces d' entreposage "
            "situées casablanca-oukacha quatre mille mètres carrés had soualem quatre mille cent mètres carrés et ain sebaâ quatorze mille mètres carrés",
        )

        self.assertEqual(format_text_latin("Had Soualem (4 100 m²) et Ain Sebaâ", remove_parenthesis=True), "had soualem et ain sebaâ")

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
            "attentats du vingt et un juillet le suspect extradé d' italie inculpé à londres nous sommes le premier janvier deux mille dix-neuf à dix heures il fait moins un degré celsius",
        )

        self.assertEqual(
            format_text_latin("31/01/2019 et 2019/01/01"),
            "trente et un janvier deux mille dix-neuf et premier janvier deux mille dix-neuf",
        )

        self.assertEqual(format_text_latin("32/01/2019 31/13 "), "trente-deux zéro un deux mille dix-neuf trente et un treizièmes")

        self.assertEqual(
            format_text_latin("Une quantité c' est un volume  Quand on divise un volume par une surface on obtient une hauteur Par exemple un 1 litre / m² = 1 000 000 mm3/1 000 000 mm² = 1mm"),
            "une quantité c' est un volume quand on divise un volume par une surface on obtient une hauteur par exemple un un litre par mètre carré égale un million millimètres cubes un million millimètres carrés égale un millimètre",
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
            "$5, 12,5 € et 1,5 M€": "cinq dollars douze virgule cinq euros et un virgule cinq million d' euros",
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
        # With a capital letter at the beginning of a sentence
        for text, expected in {
            "M. Dupont et M. Durand": "Monsieur Dupont et monsieur Durand",
            "Avec Mme Dupont. Mlle Durand aussi": "Avec madame Dupont. Mademoiselle Durand aussi",
            "Non ! Dr. Martin": "Non! Docteur Martin",
            "« M. Dupont », oui, M. Durand": "Monsieur Dupont, oui, monsieur Durand",
            "Alors... M. Dupont": "Alors... monsieur Dupont",
        }.items():
            self.assertEqual(format_text_latin(text, lower_case=False, keep_punc=True), expected)

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

    def test_all_units(self):
        # Every symbol, in every language
        for lang, units in _UNITS.items():
            two, one = {
                "fr": ("deux", "un"),
                "en": ("two", "one"),
                "es": ("dos", "un"),
                "de": ("zwei", "ein"),
                "it": ("due", "un"),
                "pt": ("dois", "um"),
                "nl": ("twee", "één"),
            }[lang]
            for symbols, (singular, plural) in units.items():
                for symbol in symbols.split("|"):
                    for text in [f"2 {symbol}", f"2{symbol}", f"2 {symbol}."]:
                        self.assertEqual(format_text_latin(text, lang=lang), f"{two} {plural.lower()}", msg=f"{lang} {text}")
                    said = format_text_latin(f"1 {symbol}", lang=lang).split(" ", 1)
                    self.assertEqual(said[1], singular.lower(), msg=f"{lang} 1 {symbol}")
                    self.assertIn(said[0], [one, one + "e", one + "a"], msg=f"{lang} 1 {symbol}")

    def test_units_squares(self):
        for lang, expected in {
            "fr": "deux mètres carrés trois kilomètres carrés un mètre cube un centimètre carré deux millimètres cubes",
            "en": "two square meters three square kilometers one cubic meter one square centimeter two cubic millimeters",
            "es": "dos metros cuadrados tres kilómetros cuadrados un metro cúbico un centímetro cuadrado dos milímetros cúbicos",
            "de": "zwei quadratmeter drei quadratkilometer ein kubikmeter ein quadratzentimeter zwei kubikmillimeter",
            "it": "due metri quadrati tre chilometri quadrati un metro cubo un centimetro quadrato due millimetri cubi",
            "pt": "dois metros quadrados três quilómetros quadrados um metro cúbico um centímetro quadrado dois milímetros cúbicos",
            "nl": "twee vierkante meter drie vierkante kilometer één kubieke meter één vierkante centimeter twee kubieke millimeter",
        }.items():
            for text in "2 m² 3 km² 1 m³ 1 cm² 2 mm³", "2 m2 3 km2 1 m3 1 cm2 2 mm3":
                self.assertEqual(format_text_latin(text, lang=lang), expected)

    def test_units_ratios(self):
        for lang, texts in {
            "fr": {
                "3 mg/j": "trois milligrammes par jour",
                "3 mg/jour": "trois milligrammes par jour",
                "3 mg / jour": "trois milligrammes par jour",
                "1 mg/j": "un milligramme par jour",
                "500 mg/kg": "cinq cents milligrammes par kilogramme",
                "1,5 mg/kg/j": "un virgule cinq milligramme par kilogramme par jour",
                "1/2 mg/j": "un demi milligramme par jour",
                "10 ml/h et 5 g/l": "dix millilitres par heure et cinq grammes par litre",
                "3 mg/dl, 2 l/min": "trois milligrammes par décilitre deux litres par minute",
                "50 µg/h": "cinquante microgrammes par heure",
                "10 m/s, 5 km/h": "dix mètres par seconde cinq kilomètres par heure",
                "3 fois/j, 3x/j": "trois fois par jour trois fois par jour",
                "1 cp/j, 2 cp / jour": "un comprimé par jour deux comprimés par jour",
                "2 gouttes/jour, 3 sachets/semaine": "deux gouttes par jour trois sachets par semaine",
                "35h/semaine, 1 h/j": "trente-cinq heures par semaine une heure par jour",
                "5 €/m²": "cinq euros par mètre carré",
                "1 €/kg et 1,20 €/kg": "un euro par kilogramme et un euro vingt par kilogramme",
                "24h/24, 24 h/24h, 24 heures/24": "vingt-quatre heures sur vingt-quatre " * 2 + "vingt-quatre heures sur vingt-quatre",
                "7j/7 et 6 jours/7": "sept jours sur sept et six jours sur sept",
                # Not read as "par"
                "et/ou, matin/soir": "et ou matin soir",
                "100 mg/5 ml": "cent milligrammes cinq millilitres",
                "100 mg/200 mg": "cent milligrammes deux cents milligrammes",
                "1 g/24h": "un gramme vingt-quatre heures",
                "3 mg/xyz, 5 m/x": "trois milligrammes xyz cinq m x",
                "12/8 et 1/2": "douze huitièmes et un demi",
                "mg/j": "milligrammes j",
            },
            "en": {
                "5 mg/day": "five milligrams per day",
                "1 mg/d": "one milligram per day",
                "5 mg/kg/day": "five milligrams per kilogram per day",
                "100 ml/h, 100 ml/hr": "one hundred milliliters per hour one hundred milliliters per hour",
                "5 g/l and 5 m/s": "five grams per liter and five meters per second",
                "60 mi/h, 10 km/h": "sixty miles per hour ten kilometers per hour",
                "3 times/day, 2x/day, 1 tablet/day": "three times per day two times per day one tablet per day",
                "1,200 kg/m³": "one thousand two hundred kilograms per cubic meter",
                "$5/share and $5M/year": "five dollars per share and five million dollars per year",
                "5 €/m²": "five euros per square meter",
                # Not read as "per"
                "and/or, A/B": "and or a b",
                "24/7": "twenty-four seven",
                "100 mg/5 ml": "one hundred milligrams five milliliters",
                "a 5 g network, 5 m people": "a five g network five m people",
            },
            "es": {
                "3 mg/día": "tres miligramos por día",
                "5 km/h": "cinco kilómetros por hora",
                "500 mg/kg": "quinientos miligramos por kilogramo",
                "3 veces/día, 1 h/día": "tres veces por día una hora por día",
                "5 €/m²": "cinco euros por metro cuadrado",
                "y/o": "y o",
            },
            "de": {
                "3 mg/Tag": "drei milligramm pro tag",
                "5 km/h": "fünf kilometer pro stunde",
                "500 mg/kg": "fünfhundert milligramm pro kilogramm",
                "3x/Tag, 1 h/Tag": "drei mal pro tag eine stunde pro tag",
                "5 €/m²": "fünf euro pro quadratmeter",
                "2,1 g": "zwei komma eins gramm",
                "und/oder": "und oder",
            },
            "it": {
                "3 mg/die, 3 mg/giorno": "tre milligrammi al giorno tre milligrammi al giorno",
                "5 km/h": "cinque chilometri all' ora",
                "500 mg/kg": "cinquecento milligrammi per chilogrammo",
                "3 volte/giorno": "tre volte al giorno",
                "5 €/kg": "cinque euro per chilogrammo",
                "24h/24, 7 giorni/7": "ventiquattro ore su ventiquattro sette giorni su sette",
                "e/o": "e o",
            },
            "pt": {
                "3 mg/dia": "três miligramas por dia",
                "5 km/h": "cinco quilómetros por hora",
                "500 mg/kg": "quinhentos miligramas por quilograma",
                "3 vezes/dia, 1 h/dia": "três vezes por dia uma hora por dia",
                "5 €/m²": "cinco euros por metro quadrado",
                "e/ou": "e ou",
            },
            "nl": {
                "3 mg/dag": "drie milligram per dag",
                "5 km/h, 5 km/u": "vijf kilometer per uur vijf kilometer per uur",
                "500 mg/kg": "vijfhonderd milligram per kilogram",
                "3x/dag": "drie keer per dag",
                "5 €/m²": "vijf euro per vierkante meter",
                "en/of": "en of",
            },
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected, msg=f"{lang} {text}")

    def test_slash_with_punctuation(self):
        for lang, texts in {
            "fr": {
                "Il prend 3 mg/j, et/ou 5 ml.": "Il prend trois milligrammes par jour, et ou cinq millilitres.",
                "2,5mg/j.": "Deux virgule cinq milligrammes par jour.",
                "matin/soir": "matin soir",
            },
            "en": {"He takes 5 mg/day, and/or 2 ml.": "He takes five milligrams per day, and or two milliliters."},
            "de": {"3 mg/Tag.": "Drei Milligramm pro Tag."},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang, keep_punc=True, lower_case=False), expected)
        # Numbers not converted
        self.assertEqual(format_text_latin("3 mg/j et/ou 1/2", keep_punc=True, convert_numbers=False), "3 mg j et ou 1 2")

    def test_singular_below_two(self):
        # French: the plural starts at two
        for text, expected in {
            "0 km, 0,5 km, 1 km, 1,5 km, 1,99 km": "zéro kilomètre zéro virgule cinq kilomètre un kilomètre un virgule cinq kilomètre un virgule quatre-vingt-dix-neuf kilomètre",
            "2 km, 2,0 km, 11 km, 21 km, 1 500 km": "deux kilomètres deux virgule zéro kilomètres onze kilomètres vingt et un kilomètres mille cinq cents kilomètres",
            "1.5 kg, 1.000 g": "un point cinq kilogramme mille grammes",
            "1,5 mg/j, 0,5 µg/h": "un virgule cinq milligramme par jour zéro virgule cinq microgramme par heure",
            "1,5 m²": "un virgule cinq mètre carré",
            "1,20 €, 0,50 €, 1,5 €, 2,20 €": "un euro vingt zéro euro cinquante un virgule cinq euro deux euros vingt",
            "$1,50, $1,5, $2,50": "un dollar cinquante un virgule cinq dollar deux dollars cinquante",
            "1 M€, 1,5 M€, 2,5 M€": "un million d' euros un virgule cinq million d' euros deux virgule cinq millions d' euros",
            "1,5 Md€, 2 Md€": "un virgule cinq milliard d' euros deux milliards d' euros",
            "-1 °C, 1,5 °C, 2 °C, 1 °F, 1°": "moins un degré celsius un virgule cinq degré celsius deux degrés celsius un degré fahrenheit un degré",
        }.items():
            self.assertEqual(format_text_latin(text), expected)
        # English: only for one
        self.assertEqual(
            format_text_latin("0 km, 0.5 km, 1 km, 1.5 km, 1 °C, 1.5 °C", lang="en"),
            "zero kilometers zero point five kilometers one kilometer one point five kilometers one degree celsius one point five degrees celsius",
        )
        # The same sound, in the WER
        for ref, pred in [("1,5 mg", "un virgule cinq milligrammes"), ("1,5 M€", "un virgule cinq millions d'euros"), ("1,20 €", "un euros vingt"), ("1 °C", "un degrés celsius")]:
            self.assertEqual(compute_wer([ref], [pred], language="fr", remove_apostrophes_and_dashes=True)["wer"], 0, msg=f"{ref} / {pred}")

    def test_one_agreement(self):
        for lang, texts in {
            "fr": {"1 min, 1 goutte/jour, 1 £": "une minute une goutte par jour une livre", "un livre, un heure": "un livre un heure"},
            "es": {"1 h/día, 1 ha, 1 km": "una hora por día una hectárea un kilómetro", "uno kilómetro": "uno kilómetro"},
            "it": {"1 volta/giorno, 1 km": "una volta al giorno un chilometro"},
            "de": {"1 min, 1 km, 21 km": "eine minute ein kilometer einundzwanzig kilometer"},
            "pt": {"1 gota/dia, 1 km": "uma gota por dia um quilómetro"},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(format_text_latin(text, lang=lang), expected)
        self.assertEqual(numbers_and_symbols_to_letters("1 €.", lang="de"), "Ein Euro.")
        self.assertEqual(numbers_and_symbols_to_letters("$1", lang="es"), "un dólar")

    def test_one_currency(self):
        for lang, expected in {
            "fr": "un euro un dollar une livre vingt et un euros mille un euros",
            "en": "one euro one dollar one pound twenty-one euros one thousand and one euros",
            "es": "un euro un dólar una libra veintiuno euros mil uno euros",
            "de": "ein euro ein dollar ein pfund einundzwanzig euro eintausendeins euro",
            "it": "un euro un dollaro una sterlina ventuno euro milleuno euro",
            "pt": "um euro um dólar uma libra vinte e um euros mil e um euros",
            "nl": "één euro één dollar één pond eenentwintig euro duizendéén euro",
        }.items():
            for text in "1 €, 1 $, 1 £, 21 €, 1 001 €", "€1, $1, £1, €21, 1 001 €":
                self.assertEqual(format_text_latin(text, lang=lang), expected, msg=f"{lang} {text}")

    def test_wer_units(self):
        for lang, texts in {
            "fr": [
                ("3 mg/j", "trois milligrammes par jour"),
                ("1 mg", "un milligramme"),
                ("1 mg", "un milligrammes"),
                ("0,5 mg", "zéro virgule cinq milligramme"),
                ("5 µg", "cinq micro grammes"),
                ("5 m²", "cinq mètre carré"),
                ("1 €", "un euro"),
                ("et/ou", "et ou"),
                ("24h/24", "vingt-quatre heures sur vingt-quatre"),
            ],
            "en": [("5 mg/day", "five milligrams per day"), ("1 mg", "one milligram"), ("and/or", "and or")],
            "es": [("3 mg/día", "tres miligramos por día"), ("1 km", "un kilómetro")],
            "de": [("3 mg/Tag", "drei Milligramm pro Tag"), ("1 km", "ein Kilometer")],
            "it": [("3 mg/die", "tre milligrammi al giorno"), ("1 km", "un chilometro")],
            "pt": [("3 mg/dia", "três miligramas por dia"), ("1 km", "um quilômetro")],
            "nl": [("3 mg/dag", "drie milligram per dag")],
        }.items():
            for ref, pred in texts:
                for remove in [False, True]:
                    self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=remove)["wer"], 0, msg=f"{ref} / {pred} ({remove})")
        for lang, ref, pred in [("fr", "3 mg/j", "trois grammes par jour"), ("fr", "3 mg/j", "trois milligrammes par heure"), ("en", "5 mg/day", "five milligrams per hour")]:
            self.assertGreater(compute_wer([ref], [pred], language=lang)["wer"], 0, msg=f"{ref} / {pred}")

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

    def test_acronyms_without_punctuation(self):
        # The dots of the acronyms do not split them, whatever is kept
        for lang, texts in {
            "fr": {
                "La S.N.C.F. a dit": "la sncf a dit",
                "la S.N.C.F, oui": "la sncf oui",
                "la (S.N.C.F.).": "la sncf",
                "l'O.N.U. et l'U.E.": "l' onu et l' ue",
                "À la S.N.C.F.": "à la sncf",
                "50,0 U.I. par jour": "cinquante virgule zéro ui par jour",
            },
            "en": {"The U.S. and the U.K. are": "the us and the uk are", "in the U.S.A.": "in the usa"},
            "es": {"la O.N.U. dijo": "la onu dijo"},
            "de": {"die U.S.A. sind": "die usa sind"},
        }.items():
            for text, expected in texts.items():
                self.assertEqual(normalize_text(text, lang), expected)
                self.assertEqual(normalize_text(text, lang, for_wer=True), expected)
                self.assertEqual(normalize_text(text, lang, for_wer=True, remove_apostrophes_and_dashes=True), expected.replace("'", ""))
        self.assertEqual(normalize_text("La S.N.C.F. a dit", "fr", case_sensitive=True), "La SNCF a dit")
        self.assertEqual(normalize_text("La S.N.C.F. a dit.", "fr", case_sensitive=True, keep_punctuation=True), "La SNCF a dit.")
        for lang, ref, pred in [("fr", "La S.N.C.F. a dit", "La SNCF a dit"), ("en", "The U.S. is", "The US is")]:
            for kwargs in [{}, {"case_sensitive": True}, {"keep_punctuation": True}, {"remove_apostrophes_and_dashes": True}]:
                self.assertEqual(compute_wer([ref], [pred], language=lang, **kwargs)["wer"], 0, msg=f"{ref} / {pred} ({kwargs})")
            self.assertGreater(compute_wer([ref], [pred.replace("US", "U S").replace("SNCF", "S N C F")], language=lang)["wer"], 0)
            self.assertGreater(compute_wer([ref], [pred.lower()], language=lang, case_sensitive=True)["wer"], 0)

    def test_normalize_text(self):
        text = "Euh, Nicotine 1,5 mg. L'infirmière prend une semaine au rendez-vous, à 14:30 !"
        for kwargs, expected in [
            ({"keep_disfluencies": True}, "euh nicotine un virgule cinq milligramme l' infirmière prend une semaine au rendez-vous à quatorze heures trente"),
            ({"case_sensitive": True, "keep_disfluencies": True}, "Euh Nicotine un virgule cinq milligramme L' infirmière prend une semaine au rendez-vous à quatorze heures trente"),
            ({}, "nicotine un virgule cinq milligramme l' infirmière prend une semaine au rendez-vous à quatorze heures trente"),
            ({"for_wer": True}, "nicotine un virgule cinq milligramme l' infirmière prend un semaine au rendez-vous à quatorze heure trente"),
            ({"for_wer": True, "keep_disfluencies": True}, "euh nicotine un virgule cinq milligramme l' infirmière prend un semaine au rendez-vous à quatorze heure trente"),
            (
                {"for_wer": True, "remove_apostrophes_and_dashes": True},
                "nicotine un virgule cinq milligramme l infirmière prend un semaine au rendez vous à quatorze heure trente",
            ),
            (
                {"for_wer": True, "case_sensitive": True, "remove_apostrophes_and_dashes": True},
                "Nicotine un virgule cinq milligramme L infirmière prend un semaine au rendez vous à quatorze heure trente",
            ),
            (
                {"for_wer": True, "case_sensitive": True, "keep_punctuation": True},
                "Nicotine un virgule cinq milligramme . L' infirmière prend un semaine au rendez-vous , à quatorze heure trente !",
            ),
            (
                {"for_wer": True, "keep_punctuation": True, "remove_apostrophes_and_dashes": True},
                "nicotine un virgule cinq milligramme . l infirmière prend un semaine au rendez vous , à quatorze heure trente !",
            ),
        ]:
            self.assertEqual(normalize_text(text, "fr", **kwargs), expected, msg=f"{kwargs}")
        self.assertEqual(normalize_text("les chats noirs", "fr", ignore_final_s=True), "le chat noir")
        self.assertEqual(normalize_text("Two hundred and fifty", "en"), "two hundred and fifty")
        self.assertEqual(normalize_text("Two hundred and fifty", "en-US", for_wer=True), "two hundred fifty")
        self.assertEqual(normalize_text("Тест №1, 36%", "ru"), normalize_text("Тест №1, 36%", "ru", for_wer=True))
        self.assertEqual(normalize_text("يقول Hello", "ar"), "يقول hello")
        self.assertEqual(normalize_text("يقول Hello", "ar", case_sensitive=True), "يقول Hello")
        with self.assertRaises(NotImplementedError):
            normalize_text("a", "xx")

    def test_wer_interface(self):
        def wer(ref, pred, **kwargs):
            return compute_wer([ref], [pred], **kwargs)["wer"]

        ref = "L'état, le rendez-vous"
        # Case and punctuation do not matter by default
        self.assertEqual(wer(ref, "l'état le rendez-vous", language="fr"), 0)
        self.assertGreater(wer(ref, "l'état, le rendez-vous", language="fr", case_sensitive=True), 0)
        self.assertEqual(wer(ref, "L'état le rendez-vous", language="fr", case_sensitive=True), 0)
        self.assertGreater(wer(ref, "L'état le rendez-vous", language="fr", keep_punctuation=True), 0)
        self.assertEqual(wer(ref, "l'état , le rendez-vous", language="fr", keep_punctuation=True), 0)
        # Each punctuation mark is a word
        self.assertEqual(compute_wer(["oui, non."], ["oui non"], language="fr", keep_punctuation=True)["wer"], 0.5)
        # Apostrophes and dashes
        self.assertGreater(wer(ref, "l état le rendez vous", language="fr"), 0)
        self.assertEqual(wer(ref, "l état le rendez vous", language="fr", remove_apostrophes_and_dashes=True), 0)
        # Final s
        self.assertGreater(wer("les chats", "le chat", language="fr"), 0)
        self.assertEqual(wer("les chats", "le chat", language="fr", ignore_final_s=True), 0)
        # Punctuation removed after the numbers
        for kwargs in [{}, {"case_sensitive": True}]:
            self.assertEqual(wer("Il prend 1,5 mg.", "Il prend un virgule cinq milligramme", language="fr", **kwargs), 0)
        # No language: the texts as they are
        self.assertGreater(wer("Oui.", "oui"), 0)

        # normalize_text does what compute_wer does
        texts = ["Euh, l'état prend 1,5 mg. Une femme, la S.N.C.F. !", "Two"]
        for kwargs in [{}, {"case_sensitive": True}, {"keep_punctuation": True}, {"remove_apostrophes_and_dashes": True}, {"ignore_final_s": True}, {"keep_disfluencies": True}]:
            out = compute_wer(texts, texts, language="fr", alignment=os.devnull, **kwargs)["raw_alignement"]
            self.assertEqual([" ".join(words) for words in out.references], [normalize_text(text, "fr", for_wer=True, **kwargs) for text in texts])

    def test_wer_deprecated_normalization(self):
        texts = ["Euh, l'état prend 1,5 mg. Les rendez-vous, la S.N.C.F. !", "Une femme"]
        preds = ["L état prend 1,5 mg, le rendez vous", "un femme"]
        for normalization, kwargs in {
            "fr": {"language": "fr"},
            "fr+": {"language": "fr", "remove_apostrophes_and_dashes": True},
            "fr++": {"language": "fr", "remove_apostrophes_and_dashes": True, "ignore_final_s": True},
        }.items():
            with self.assertWarns(FutureWarning) as warning:
                old = compute_wer(texts, preds, normalization=normalization)
            self.assertIn("deprecated", str(warning.warning))
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                self.assertEqual(old, compute_wer(texts, preds, **kwargs))
        with self.assertRaises(ValueError):
            compute_wer(texts, preds, normalization="fr", language="fr")

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
        for ref, pred, lang in [("euh le matin", "le matin hum", "fr"), ("uh yes", "yes um", "en")]:
            self.assertEqual(compute_wer([ref], [pred], language=lang)["wer"], 0)
            self.assertGreater(compute_wer([ref], [pred], language=lang, keep_disfluencies=True)["wer"], 0)
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
                for remove in [False, True]:
                    self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=remove)["wer"], 0, msg=f"{ref} / {pred} ({remove})")
        # Numbers that are not years, not changed
        self.assertEqual(normalize_spoken_numbers("one hundred twenty twenty one", "en"), "one hundred twenty twenty one")
        self.assertEqual(normalize_spoken_numbers("twenty twenty thousand", "en"), "twenty twenty thousand")
        # What was said is different
        for ref, pred in [("don't", "do not"), ("gonna", "going to"), ("in 1789", "in 1798")]:
            self.assertGreater(compute_wer([ref], [pred], language="en", remove_apostrophes_and_dashes=True)["wer"], 0)

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
            self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=True)["wer"], 0)

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
        # A number alone can be in the middle of a text converted word by word
        for text, expected in {"3.": "trois.", "2024,": "deux mille vingt-quatre,", "12?": "douze?", "5€.": "cinq euros."}.items():
            self.assertEqual(numbers_and_symbols_to_letters(text, lang="fr"), expected)
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
            self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=True)["wer"], 0, msg=f"{ref} / {pred}")
        for lang, ref, pred in [("fr", "0,025", "zéro virgule deux cinq"), ("en", "7 October", "eighth October"), ("nl", "1905", "negentienhonderdzes")]:
            self.assertGreater(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=True)["wer"], 0, msg=f"{ref} / {pred}")

    def test_wer_parentheses(self):
        # The words in parentheses are kept
        for lang, ref, pred, wrong in [
            ("fr", "le matin (et le soir) oui", "le matin et le soir oui", "le matin oui"),
            ("en", "the morning (and evening) yes", "the morning and evening yes", "the morning yes"),
            ("ru", "утром (и вечером) да", "утром и вечером да", "утром да"),
        ]:
            for remove in [False, True]:
                self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=remove)["wer"], 0, msg=f"{ref} ({remove})")
                self.assertGreater(compute_wer([ref], [wrong], language=lang, remove_apostrophes_and_dashes=remove)["wer"], 0, msg=f"{ref} ({remove})")

    def test_wer_arabic_case(self):
        # The case of the Latin words does not count in the WER
        self.assertEqual(compute_wer(["يقول Hello"], ["يقول hello"], language="ar")["wer"], 0)

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
                for remove in [False, True]:
                    self.assertEqual(compute_wer([ref], [pred], language=lang, remove_apostrophes_and_dashes=remove)["wer"], 0, msg=f"{ref} / {pred} ({remove})")
            self.assertGreater(compute_wer(["1,5 et 99"], ["15 et 99"], language=lang)["wer"], 0)
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
                "ça coute un euro vingt",
            )

        for text in [
            "ça coute 1,20$",
            "ça coute 1,20 $",
        ]:
            self.assertEqual(
                format_text_latin(text),
                "ça coute un dollar vingt",  # TODO: remove the s...
            )

        for text in [
            "ça coute 1.20$ * 6 mois",
            "ça coute 1.20 $ * 6 mois",
        ]:
            self.assertEqual(format_text_latin(text), "ça coute un point vingt dollar six mois")

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


class TestWer(Test):
    """
    The expected values of test_no_regression_* are the ones of the code before the options language, case_sensitive, ...
    (except the forms of the alef in Arabic, merged since)
    """

    pairs_fr = [
        ("Le chat, la souris et les chiens.", "le chat la souris et les chien"),
        ("C'est l'état d'hier !", "c'est l état hier"),
        ("Un rendez-vous important.", "un rendez vous important"),
        ("Les garçons mangent 3 pommes", "le garçon mange trois pommes"),
        ("Oui", "oui non peut-être"),
        ("Il pleut beaucoup ici", "il pleut"),
        ("Œuvre, cœur et garçon", "oeuvre coeur garçon"),
        ("Une femme arrive", "un femme arrive"),
    ]
    pairs_ar = [
        ("مرحبا بكم في المدينة", "مرحبا بكم المدينة"),
        ("كِتَابٌ جَدِيد", "كتاب جديد"),
        ("نص بسيط don't high-quality", "نص بسيط don t high quality"),
        ("الخبزة، والماء؟", "الخبزة والماء"),
        ("يوجد 10 أشخاص", "يوجد عشرة أشخاص هنا"),
    ]

    # Previous syntax -> new syntax
    syntaxes = {
        "fr": {"language": "fr"},
        "fr+": {"language": "fr", "remove_apostrophes_and_dashes": True},
        "fr++": {"language": "fr", "remove_apostrophes_and_dashes": True, "ignore_final_s": True},
        "ar": {"language": "ar"},
        "ar+": {"language": "ar", "remove_apostrophes_and_dashes": True},
    }

    all_options = [
        dict(case_sensitive=c, keep_punctuation=p, remove_apostrophes_and_dashes=r, keep_disfluencies=d, ignore_final_s=s)
        for c in [False, True]
        for p in [False, True]
        for r in [False, True]
        for d in [False, True]
        for s in [False, True]
    ]

    def setUp(self):
        super().setUp()
        warnings.simplefilter("ignore", FutureWarning)

    def tearDown(self):
        warnings.resetwarnings()
        super().tearDown()

    def compared_texts(self, texts, **kwargs):
        """The texts as they are compared in compute_wer"""
        out = compute_wer(texts, texts, alignment=os.devnull, **kwargs)["raw_alignement"]
        return [" ".join(words) for words in out.references]

    def errors(self, ref, pred, **kwargs):
        """(substitutions, deletions, insertions, hits, number of words of the reference)"""
        refs, preds = ([ref], [pred]) if isinstance(ref, str) else (ref, pred)
        res = compute_wer(refs, preds, **kwargs)
        return tuple(round(res[k] * res["count"]) for k in ["sub", "del", "ins", "hits"]) + (res["count"],)

    def test_no_regression_texts(self):
        for lang, texts in {
            "fr": {
                "Le chat, la souris et les chiens.": [
                    "le chat la souris et les chiens",
                    "le chat la souris et les chiens",
                    "le chat la souri et le chien",
                ],
                "C'est l'état d'aujourd'hui !": [
                    "c' est l' état d' aujourd'hui",
                    "c est l état d aujourd hui",
                    "c est l état d aujourd hui",
                ],
                "Œuvre, cœur et garçon": [
                    "oeuvre coeur et garçon",
                    "oeuvre coeur et garçon",
                    "oeuvre coeur et garçon",
                ],
                "Les 20 000 personnes": [
                    "les vingt mille personnes",
                    "les vingt mille personnes",
                    "le vingt mille personne",
                ],
                "des mots déjà normalisés": [
                    "des mots déjà normalisés",
                    "des mots déjà normalisés",
                    "de mot déjà normalisé",
                ],
                "2ème étage": [
                    "deuxième étage",
                    "deuxième étage",
                    "deuxième étage",
                ],
                "www.linagora.com": [
                    "www point linagora point com",
                    "www point linagora point com",
                    "www point linagora point com",
                ],
            },
            "ar": {
                "مرحبا بكم": [
                    "مرحبا بكم",
                    "مرحبا بكم",
                ],
                "في اللغة الإنجليزية ، يمكن للمرء أن يقول \"Hello world\"!": [
                    "في اللغة الانجليزية يمكن للمرء ان يقول hello world",
                    "في اللغة الانجليزية يمكن للمرء ان يقول hello world",
                ],
                "يوجد 10 أشخاص": [
                    "يوجد عشرة اشخاص",
                    "يوجد عشرة اشخاص",
                ],
                "كِتَابٌ جَدِيد": [
                    "كتاب جديد",
                    "كتاب جديد",
                ],
                "10 $": [
                    "عشرة دولار",
                    "عشرة دولار",
                ],
                "صببباح النووووور": [
                    "صبباح النوور",
                    "صبباح النوور",
                ],
                "بعض الأرقام: 01 و 314": [
                    "بعض الارقام صفر واحد و ثلاثمائة و اربعة عشر",
                    "بعض الارقام صفر واحد و ثلاثمائة و اربعة عشر",
                ],
                "الخبزة، والماء؟": [
                    "الخبزة والماء",
                    "الخبزة والماء",
                ],
                "2020/04/22 اليوم": [
                    "اثنان و عشرون ابريل الفان و عشرون اليوم",
                    "اثنان و عشرون ابريل الفان و عشرون اليوم",
                ],
                "نص بسيط don't high-quality": [
                    "نص بسيط don't high-quality",
                    "نص بسيط don t high quality",
                ],
            },
        }.items():
            for text, expected in texts.items():
                for suffix, e in zip(["", "+", "++"], expected):
                    self.assertEqual(self.compared_texts([text], normalization=lang + suffix), [e], msg=f"{text} ({lang + suffix})")
                    self.assertEqual(self.compared_texts([text], **self.syntaxes[lang + suffix]), [e], msg=f"{text} ({lang + suffix})")

    def test_no_regression_tunisian(self):
        for text, tunisian, arabic in [
            ("انشا الله  بسبعه عسلامة يسلمك ديجا", "ان شاء الله بسبع عالسلامة يسلمك déjà", "انشا الله بسبعه عسلامة يسلمك ديجا"),
            ("بڨداش أﻛ الخبزﻫ", "بقداش اك الخبزه", "بقداش اك الخبزه"),
        ]:
            for kwargs in [{"normalization": "ar_tn"}, {"normalization": "ar_tn+"}, {"language": "ar_tn"}, {"language": "ar_tn", "remove_apostrophes_and_dashes": True}]:
                self.assertEqual(self.compared_texts([text], **kwargs), [tunisian])
            self.assertEqual(self.compared_texts([text], language="ar"), [arabic])

    def test_changes_on_purpose(self):
        # Numbers ("une femme" -> "1 femme" before), units, acronyms, titles, spelling, disfluencies, text in parentheses
        for lang, texts in {
            "fr": {
                "Un rendez-vous à Aix-en-Provence": [
                    "un rendez-vous à aix-en-provence",
                    "un rendez vous à aix en provence",
                    "un rendez vou à aix en provence",
                ],
                "Une femme et un homme": [
                    "un femme et un homme",
                    "un femme et un homme",
                    "un femme et un homme",
                ],
                "Il y a 3 enfants et 21 adultes": [
                    "il y a trois enfants et vingt et un adultes",
                    "il y a trois enfants et vingt et un adultes",
                    "il y a troi enfant et vingt et un adulte",
                ],
                "En 1998, 50 % des gens": [
                    "en mille neuf cent quatre vingt dix huit cinquante pour cent des gens",
                    "en mille neuf cent quatre vingt dix huit cinquante pour cent des gens",
                    "en mille neuf cent quatre vingt dix huit cinquante pour cent de gen",
                ],
                "Ça coûte 12 € ou 1,20 €": [
                    "ça coute douze euro ou un euro vingt",
                    "ça coute douze euro ou un euro vingt",
                    "ça coute douze euro ou un euro vingt",
                ],
                "À 10h30, le 1er mai 2020": [
                    "à dix heure trente le premier mai deux mille vingt",
                    "à dix heure trente le premier mai deux mille vingt",
                    "à dix heure trente le premier mai deux mille vingt",
                ],
                "Il a dit : « Bonjour » (oui)": [
                    "il a dit bonjour oui",
                    "il a dit bonjour oui",
                    "il a dit bonjour oui",
                ],
                "Euh, je ne sais pas hum": [
                    "je ne sais pas",
                    "je ne sais pas",
                    "je ne sai pa",
                ],
                "L'événement du maître": [
                    "l' évènement du maitre",
                    "l évènement du maitre",
                    "l évènement du maitre",
                ],
                "M. Dupont et Mme Durand": [
                    "monsieur dupont et madame durand",
                    "monsieur dupont et madame durand",
                    "monsieur dupont et madame durand",
                ],
                "La S.N.C.F. et l'ONU": [
                    "la sncf et l' onu",
                    "la sncf et l onu",
                    "la sncf et l onu",
                ],
                "3 kg et 1 km": [
                    "trois kilogramme et un kilomètre",
                    "trois kilogramme et un kilomètre",
                    "troi kilogramme et un kilomètre",
                ],
                "Il a 1,5 million": [
                    "il a un virgule cinq million",
                    "il a un virgule cinq million",
                    "il a un virgule cinq million",
                ],
            },
        }.items():
            for text, expected in texts.items():
                for suffix, e in zip(["", "+", "++"], expected):
                    self.assertEqual(self.compared_texts([text], normalization=lang + suffix), [e], msg=f"{text} ({lang + suffix})")
                    self.assertEqual(self.compared_texts([text], **self.syntaxes[lang + suffix]), [e], msg=f"{text} ({lang + suffix})")

    def test_no_regression_wer(self):
        for normalization, expected in {
            # (substitutions, deletions, insertions, hits, words of the reference)
            "fr": (6, 4, 3, 23, 33),
            "fr+": (4, 4, 3, 26, 34),
            "fr++": (1, 4, 3, 29, 34),
            "ar": (2, 1, 3, 12, 15),
            "ar+": (0, 1, 1, 16, 17),
        }.items():
            refs, preds = map(list, zip(*(self.pairs_fr if normalization.startswith("fr") else self.pairs_ar)))
            self.assertEqual(self.errors(refs, preds, normalization=normalization), expected)
            self.assertEqual(self.errors(refs, preds, **self.syntaxes[normalization]), expected)
            res = compute_wer(refs, preds, normalization=normalization, use_percents=True)
            self.assertAlmostEqual(res["wer"], 100 * sum(expected[:3]) / expected[-1])
        refs, preds = map(list, zip(*self.pairs_fr))
        # Without normalization
        self.assertEqual(self.errors(refs, preds), (19, 3, 3, 9, 31))
        # Words to exclude, replacements, words to focus on
        self.assertEqual(self.errors(refs, preds, normalization="fr+", words_blacklist=["pommes"]), (1, 4, 3, 24, 29))
        self.assertEqual(self.errors(refs, preds, normalization="fr+", replacements_ref={"chat": "chien"}, replacements_pred={"non": "oui"}), (5, 4, 3, 25, 34))
        res = compute_wer(refs, preds, normalization="fr+", words_list=["Chat", "garçons", "pommes", "œuvre"])
        self.assertEqual({k: res[k] for k in ["TP", "FP", "FN"]}, {"TP": 3, "FP": 0, "FN": 1})

    def test_no_regression_cer(self):
        # (without the sentences with "un", that was "1" before)
        indices = [0, 1, 3, 4, 5, 6]
        for normalization, expected in {
            "fr": [0.03225806451612903, 0.18181818181818182, 0.125, 4.666666666666667, 0.6190476190476191, 0.13636363636363635],
            "fr+": [0.03225806451612903, 0.10526315789473684, 0.125, 4.666666666666667, 0.6190476190476191, 0.13636363636363635],
            "fr++": [0.0, 0.10526315789473684, 0.07142857142857142, 4.666666666666667, 0.6190476190476191, 0.13636363636363635],
            "ar": [0.15, 0.0, 0.07692307692307693, 0.0, 0.26666666666666666],
            "ar+": [0.15, 0.0, 0.0, 0.0, 0.26666666666666666],
        }.items():
            pairs = [self.pairs_fr[i] for i in indices] if normalization.startswith("fr") else self.pairs_ar
            for (ref, pred), e in zip(pairs, expected):
                for kwargs in [{"normalization": normalization}, self.syntaxes[normalization]]:
                    self.assertAlmostEqual(compute_wer([ref], [pred], character_level=True, **kwargs)["wer"], e, msg=f"{ref} / {pred} ({kwargs})")

    def test_option_effects(self):
        for ref, pred, options, expected in [
            # Case
            ("Le Chat noir", "le chat noir", {}, (0, 0, 0, 3, 3)),
            ("Le Chat noir", "le chat noir", {"case_sensitive": True}, (2, 0, 0, 1, 3)),
            ("DOLIPRANE", "Doliprane", {"case_sensitive": True}, (1, 0, 0, 0, 1)),
            ("Été à Évian", "été à évian", {"case_sensitive": True}, (2, 0, 0, 1, 3)),
            ("ŒUVRE", "OEUVRE", {"case_sensitive": True}, (0, 0, 0, 1, 1)),
            ("La S.N.C.F.", "La SNCF", {"case_sensitive": True}, (0, 0, 0, 2, 2)),
            # Punctuation
            ("Oui, non.", "oui non", {}, (0, 0, 0, 2, 2)),
            ("Oui, non.", "oui non", {"keep_punctuation": True}, (0, 2, 0, 2, 4)),
            ("Oui non", "oui, non.", {"keep_punctuation": True}, (0, 0, 2, 2, 2)),
            ("Oui, non.", "oui. Non.", {"keep_punctuation": True}, (1, 0, 0, 3, 4)),
            ("Oui , non !", "oui, non!", {"keep_punctuation": True}, (0, 0, 0, 4, 4)),
            ("Vraiment ?", "Vraiment ?", {"keep_punctuation": True, "case_sensitive": True}, (0, 0, 0, 2, 2)),
            ("« Oui », dit-il", '"Oui", dit-il', {"keep_punctuation": True}, (0, 0, 0, 3, 3)),
            # Punctuation that is not punctuation
            ("Il prend 1,5 mg, 3 fois.", "il prend un virgule cinq milligramme , trois fois .", {"keep_punctuation": True}, (0, 0, 0, 10, 10)),
            ("À 14:30, la S.N.C.F. dit oui.", "à quatorze heures trente, la SNCF dit oui.", {"keep_punctuation": True}, (0, 0, 0, 10, 10)),
            ("M. Dupont vient.", "Monsieur Dupont vient.", {"keep_punctuation": True, "case_sensitive": True}, (0, 0, 0, 4, 4)),
            # Apostrophes and dashes
            ("L'état-major d'hier", "l état major d hier", {}, (3, 0, 1, 1, 4)),
            ("L'état-major d'hier", "l état major d hier", {"remove_apostrophes_and_dashes": True}, (0, 0, 0, 5, 5)),
            ("l’état", "l'état", {}, (0, 0, 0, 2, 2)),
            # (the punctuation marks are still there)
            ("L'état-major, oui.", "l état major oui", {"remove_apostrophes_and_dashes": True, "keep_punctuation": True}, (0, 2, 0, 4, 6)),
            # Disfluencies
            ("Euh, oui hum", "oui", {}, (0, 0, 0, 1, 1)),
            ("Euh, oui hum", "oui", {"keep_disfluencies": True}, (0, 2, 0, 1, 3)),
            # Final s
            ("les chats noirs", "le chat noir", {}, (3, 0, 0, 0, 3)),
            ("les chats noirs", "le chat noir", {"ignore_final_s": True}, (0, 0, 0, 3, 3)),
            # Said the same way
            ("une femme, 2 hommes", "un femme deux hommes", {"case_sensitive": True}, (0, 0, 0, 4, 4)),
            # All together
            (
                "Euh, l'État-Major prend 1,5 mg.",
                "L État Major prend un virgule cinq milligrammes .",
                {"case_sensitive": True, "keep_punctuation": True, "remove_apostrophes_and_dashes": True},
                (0, 0, 0, 9, 9),
            ),
        ]:
            self.assertEqual(self.errors(ref, pred, language="fr", **options), expected, msg=f"{ref} / {pred} ({options})")

    def test_options_languages(self):
        for lang, ref, pred in [
            ("en", "Mr. Smith paid $1.50, uh, in the U.S.", "mister smith paid one dollar fifty in the us"),
            ("es", "Son 3 km, él dijo.", "son tres kilómetros él dijo"),
            ("de", "Das sind 2 km, sagte er.", "das sind zwei kilometer sagte er"),
            ("it", "Sono 3 km, disse.", "sono tre chilometri disse"),
            ("pt", "São 16 km, disse.", "são dezesseis quilômetros disse"),
            ("nl", "Het is 2 km, zei hij.", "het is twee kilometer zei hij"),
            ("ru", "Это 2 дома, да.", "это два дома да"),
            ("ar", "الخبزة، والماء Hello", "الخبزة والماء hello"),
        ]:
            self.assertEqual(self.errors(ref, pred, language=lang)[:3], (0, 0, 0), msg=f"{ref} / {pred}")
            self.assertEqual(self.errors(ref, pred, language=lang, remove_apostrophes_and_dashes=True)[:3], (0, 0, 0), msg=f"{ref} / {pred}")
            # Regional codes
            self.assertEqual(self.errors(ref, pred, language=f"{lang}-XX")[:3], (0, 0, 0), msg=f"{ref} / {pred}")
            self.assertGreater(sum(self.errors(ref, pred, language=lang, case_sensitive=True)[:3]), 0, msg=f"{ref} / {pred}")
            self.assertGreater(sum(self.errors(ref, pred, language=lang, keep_punctuation=True)[:3]), 0, msg=f"{ref} / {pred}")
            for options in self.all_options:
                self.assertEqual(self.errors(ref, ref, language=lang, **options)[:3], (0, 0, 0), msg=f"{ref} ({options})")

    def test_titles(self):
        for ref, pred in [("M. Dupont est avec Mme Durand.", "Monsieur Dupont est avec madame Durand"), ("Oui. Dr Martin", "oui. Docteur Martin")]:
            self.assertEqual(self.errors(ref.replace("Oui", "oui"), pred, language="fr", case_sensitive=True)[:3], (0, 0, 0))

    def test_aujourdhui(self):
        # A single word, anywhere in the text
        for text, expected in {
            "Aujourd'hui": "Aujourd'hui",
            "Aujourd'hui il pleut": "Aujourd'hui il pleut",
            "il pleut aujourd'hui": "il pleut aujourd'hui",
            "aujourd’hui, d'aujourd'hui.": "aujourd'hui d' aujourd'hui",
            "aujourd' hui": "aujourd'hui",
            "jusqu'aujourd'hui oui": "jusqu' aujourd'hui oui",
        }.items():
            self.assertEqual(normalize_text(text, "fr", case_sensitive=True), expected)
            self.assertEqual(normalize_text(text, "fr", case_sensitive=True, remove_apostrophes_and_dashes=True), expected.replace("'", " ").replace("  ", " "))
        self.assertEqual(self.errors("il pleut aujourd'hui", "aujourd'hui il pleut", language="fr"), (0, 1, 1, 2, 3))

    def test_same_as_normalize_text(self):
        texts = [ref for ref, _ in self.pairs_fr] + ["Euh, 1,5 mg. La S.N.C.F. (oui) !", "À 14:30, M. Dupont dit : « non »...", "3 comprimés/jour, 1 €"]
        preds = [pred for _, pred in self.pairs_fr] + ["un virgule cinq milligramme, la sncf", "à quatorze heures trente", "Trois comprimés par jour"]
        for options in self.all_options:
            refs_normalized, preds_normalized = ([normalize_text(t, "fr", for_wer=True, **options) for t in T] for T in [texts, preds])
            self.assertEqual(self.compared_texts(texts, language="fr", **options), refs_normalized)
            # compute_wer on normalized texts
            for character_level in [False, True]:
                self.assertEqual(
                    compute_wer(texts, preds, language="fr", character_level=character_level, **options),
                    compute_wer(refs_normalized, preds_normalized, character_level=character_level, case_sensitive=options["case_sensitive"]),
                    msg=f"{options}",
                )
            # Normalizing twice changes nothing
            self.assertEqual([normalize_text(t, "fr", for_wer=True, **options) for t in refs_normalized], refs_normalized, msg=f"{options}")

    def test_first_capital(self):
        # A capital letter at the beginning of the prediction is not an error when the reference has none
        for ref, pred, options, errors in [
            ("le chat noir", "Le chat noir", {}, 0),
            ("le chat noir", "Le chien noir", {}, 1),
            ("le chat noir", "Le Chat noir", {}, 1),
            ("le chat noir", "LE chat noir", {}, 1),
            ("le chat noir", "La chat noir", {}, 1),
            ("le chat noir", "Le", {}, 2),
            ("iPhone", "IPhone", {}, 0),
            ("école", "École", {}, 0),
            ("à", "À", {}, 0),
            # Only in this direction, and only the first word
            ("Le chat noir", "le chat noir", {}, 1),
            ("LE chat", "Le chat", {}, 1),
            ("le chat. Il dort", "Le chat il dort", {}, 1),
            # With the rest of the normalization
            ("l'état", "L'état", {}, 0),
            ("l'état", "L'état", {"remove_apostrophes_and_dashes": True}, 0),
            ("l'état", "L'État", {"remove_apostrophes_and_dashes": True}, 1),
            ("euh le chat", "Le chat", {}, 0),
            ("le chat", "Euh, Le chat", {}, 0),
            ("trois chats", "3 chats", {}, 0),
            ("le chat.", "Le chat.", {"keep_punctuation": True}, 0),
            ("oui, le chat", "Oui ? Le chat", {"keep_punctuation": True}, 2),
            ("le chat", "", {}, 2),
        ]:
            self.assertEqual(sum(self.errors(ref, pred, language="fr", case_sensitive=True, **options)[:3]), errors, msg=f"{ref} / {pred} ({options})")
        # In the other languages, without language, on characters
        for lang in ["en", "es", "de", "it", "pt", "nl"]:
            self.assertEqual(compute_wer(["la casa"], ["La casa"], language=lang, case_sensitive=True)["wer"], 0)
        self.assertEqual(compute_wer(["le chat"], ["Le chat"], case_sensitive=True)["wer"], 0)
        self.assertEqual(compute_wer(["le chat"], ["Le chat"])["wer"], 0.5)
        self.assertEqual(compute_wer([", le chat"], ["« Le chat"], case_sensitive=True)["sub"], 1 / 3)
        self.assertEqual(compute_wer(["le chat"], ["Le chat"], language="fr", case_sensitive=True, character_level=True)["wer"], 0)
        # Each utterance separately, and the inputs are not modified
        refs, preds = ["le chat", "Le chat"], ["Le chat", "le chat"]
        self.assertEqual(compute_wer(refs, preds, language="fr", case_sensitive=True)["wer"], 0.25)
        self.assertEqual((refs, preds), (["le chat", "Le chat"], ["Le chat", "le chat"]))

    def test_arabic_letters(self):
        # Optional marks, and forms of the alef, with and without punctuation
        for ref, pred in [
            ("أَكَلَ الخُبْزَ", "اكل الخبز"),
            ("إلى آخر ٱلكتاب", "الى اخر الكتاب"),
            ("ه\u0670ذا", "هذا"),
            ("كتـــاب", "كتاب"),
            ("يوجد 10 أشخاص", "يوجد عشرة اشخاص"),
        ]:
            for options in self.all_options:
                self.assertEqual(self.errors(ref, pred, language="ar", **options)[:3], (0, 0, 0), msg=f"{ref} / {pred} ({options})")
        # Other letters are different
        self.assertEqual(compute_wer(["على"], ["علي"], language="ar")["wer"], 1)
        # Only to compare texts, in Arabic
        self.assertEqual(normalize_text("أكل", "ar"), "أكل")
        self.assertEqual(normalize_text("أكل", "ar", for_wer=True), "اكل")
        self.assertGreater(compute_wer(["أكل"], ["اكل"])["wer"], 0)

    def test_stricter_options(self):
        # Being stricter on the case, and keeping more words, cannot remove errors
        refs, preds = map(list, zip(*self.pairs_fr))
        for options in self.all_options:
            errors = sum(self.errors(refs, preds, language="fr", **options)[:3])
            for stricter in [{"case_sensitive": True}, {"keep_punctuation": True}, {"ignore_final_s": False}]:
                self.assertGreaterEqual(sum(self.errors(refs, preds, language="fr", **{**options, **stricter})[:3]), errors, msg=f"{options} {stricter}")

    def test_empty_texts(self):
        for options in self.all_options:
            self.assertEqual(compute_wer(["", "oui"], ["", "oui"], language="fr", **options)["wer"], 0)
            self.assertEqual(self.errors(["oui", "non"], ["", "non"], language="fr", **options)[:3], (0, 1, 0))
        for empty in ["...", "!", "-", "  ", "euh", "Euh..."]:
            self.assertEqual(normalize_text(empty, "fr", for_wer=True), "", msg=empty)
            self.assertEqual(self.errors(["oui", "non"], ["oui " + empty, "non"], language="fr")[:3], (0, 0, 0), msg=empty)
            self.assertEqual(self.errors(["oui", empty], ["oui", ""], language="fr")[:3], (0, 0, 0), msg=empty)
            # Said, but nothing in the reference
            self.assertGreater(compute_wer(["oui", empty], ["oui", "non"], language="fr")["wer"], 0, msg=empty)

    def test_no_language(self):
        # The texts are compared as they are, whatever the options
        for options in self.all_options:
            self.assertEqual(self.errors("Le Chat, l'état.", "Le chat l'état.", **options), (1, 0, 0, 2, 3))
        with self.assertRaises(NotImplementedError):
            compute_wer(["a"], ["a"], language="xx")

    def test_positional_and_deprecated(self):
        ref, pred = ["L'état"], ["l état"]
        warnings.resetwarnings()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            self.assertGreater(compute_wer(ref, pred, False, "fr")["wer"], 0)
            self.assertEqual(compute_wer(ref, pred, False, "fr", False, False, True)["wer"], 0)
        for args, kwargs in [((), {"normalization": "fr+"}), ((False, "fr+"), {}), ((), {"language": "fr+"})]:
            with self.assertWarnsRegex(FutureWarning, "deprecated.*language='fr', remove_apostrophes_and_dashes=True\\)"):
                self.assertEqual(compute_wer(ref, pred, *args, **kwargs)["wer"], 0)
        # With the other options
        self.assertGreater(compute_wer(["L'état"], ["l état"], normalization="fr+", case_sensitive=True)["wer"], 0)
        with self.assertRaises(ValueError):
            compute_wer(ref, pred, language="fr", normalization="fr")

    def test_words_list(self):
        refs, preds = ["La S.N.C.F. et le Doliprane 1,5 mg", "un doliprane"], ["La SNCF et le doliprane 1,5 mg", "un Doliprane"]
        for options, expected in [
            ({}, {"TP": 3, "FP": 0, "FN": 0}),
            ({"case_sensitive": True}, {"TP": 1, "FP": 1, "FN": 1}),
        ]:
            res = compute_wer(refs, preds, language="fr", words_list=["S.N.C.F.", "Doliprane"], **options)
            self.assertEqual({k: res[k] for k in expected}, expected)
        # The inputs are not modified
        words = ["S.N.C.F."]
        compute_wer(refs, preds, language="fr", words_list=words, replacements_ref={"La": "Le"})
        self.assertEqual((refs[0], words), ("La S.N.C.F. et le Doliprane 1,5 mg", ["S.N.C.F."]))

    def test_inputs_and_outputs(self):
        options = {"language": "fr", "case_sensitive": True, "remove_apostrophes_and_dashes": True}
        expected = compute_wer(["L'état, oui.", "Non"], ["L état oui", "non"], **options)
        self.assertEqual(expected["wer"], 1 / 4)
        # Dictionaries, in any order
        self.assertEqual(compute_wer({"a": "L'état, oui.", "b": "Non"}, {"b": "non", "a": "L état oui"}, **options), expected)
        # Files
        files = {name: self.get_temp_path(name) for name in ["wer_ref.txt", "wer_pred.txt", "wer_alignment.txt"]}
        ref, pred, alignment = files.values()
        for with_ids in [False, True]:
            for f, lines in [(ref, ["L'état, oui.", "Non"]), (pred, ["L état oui", "non"])]:
                with open(f, "w", encoding="utf-8") as fid:
                    fid.write("".join(f"id{i} {line}\n" if with_ids else line + "\n" for i, line in enumerate(lines)))
            self.assertEqual(compute_wer(ref, pred, use_ids=with_ids, **options), expected)
        # Command line (with ids)
        for args, wer in [
            ([], "83.33"),
            (["--language", "fr"], "25.00"),
            (["--language", "fr", "--remove_apostrophes_and_dashes"], "0.00"),
            (["--language", "fr", "--remove_apostrophes_and_dashes", "--case_sensitive"], "25.00"),
            (["--language", "fr", "--remove_apostrophes_and_dashes", "--keep_punctuation"], "33.33"),
            (["--norm", "fr+"], "0.00"),
        ]:
            stdout = self.assertRun([self.get_lib_path("utils/wer.py"), ref, pred, *args])
            self.assertIn("WER: ", stdout)
            if args:
                self.assertIn(f"WER: {wer} %", stdout, msg=f"{args}")
        # Alignment in a file, percents, confidence intervals
        compute_wer(["Le Chat"], ["le chat"], alignment=alignment, **options)
        with open(alignment, encoding="utf-8") as fid:
            self.assertIn("Chat", fid.read())
        self.assertEqual(compute_wer(["Le chat"], ["le chat"], use_percents=True, **options)["wer"], 50)
        self.assertEqual(compute_wer(["Le chat"] * 3, ["le chat"] * 3, bootstrapping=True, **options)["wer"], 0.5)
        for f in files.values():
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
