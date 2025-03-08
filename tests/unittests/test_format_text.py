import os

from ssak.utils.text import format_text_ar, format_text_latin, format_text_ru

from .utils import Test


class TestFormatTextLatin(Test):
    def test_format_digits(self):
        self.assertEqual(
            format_text_latin("Ma grand-mère version 2.0 ne connaît pas wav2vec. Commande1"),
            "ma grand-mère version deux point zéro ne connaît pas wav to vec commande un",
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
            "une quantité c' est un volume quand on divise un volume par une surface on obtient une hauteur par exemple un un litre mètres carrés un million millimètres trois un million millimètres carrés un millimètres",
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

        self.assertEqual(format_text_ar("اليوم 22/04/2002", lang="ar_tn"), "اليوم ثنين و عشرين أفريل ألفين و ثنين")

    def test_biggest_numbers(self):
        self.assertEqual(
            format_text_ar("-123456789123456789128942"),
            "سالب مائة و ثلاثة و عشرون سكستيليونا و أربعمائة و ستة و خمسون كوينتليونا و سبعمائة و تسعة و ثمانون كوادريليونا و مائة و ثلاثة و عشرون تريليونا و"
            " أربعمائة و ستة و خمسون مليارا و سبعمائة و تسعة و ثمانون مليونا و مائة و ثمانية و عشرون ألفا و تسعمائة و اثنان و أربعون",
        )

        self.assertEqual(
            format_text_ar("-123456789123456789128942", lang="ar_tn"),
            "ناقس مياة و ثلاث و عشرين سكستيليون و أربعمياة و ستة و خمسين كوينتليون و سبعمياة و تسع و ثمانين كوادريليون و مياة و ثلاث و"
            " عشرين تريليون و أربعمياة و ستة و خمسين مليار و سبعمياة و تسع و ثمانين مليون و مياة و ثمني و عشرين ألف و تسعمياة و ثنين و أربعين",
        )

        self.assertEqual(
            format_text_ar("-1234567891234567891289425"),
            "سالب سبتيليون و مئتان و أربعة و ثلاثون سكستيليونا و خمسمائة و سبعة و ستون كوينتليونا و ثمانمائة و واحد و تسعون كوادريليونا و مئتان و أربعة و"
            " ثلاثون تريليونا و خمسمائة و سبعة و ستون مليارا و ثمانمائة و واحد و تسعون مليونا و مئتان و تسعة و ثمانون ألفا و أربعمائة و خمسة و عشرون",
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
