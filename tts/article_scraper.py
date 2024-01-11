from typing import List, Tuple, Any
from bs4 import BeautifulSoup
from charset_normalizer import detect as charset_detect
import requests
import html
import markdown
import re
import trafilatura as tr
from logger import get_logger

logger = get_logger(__name__)


def fix_encoding(text: str):
    """
    Fix utf-8 encoding errors in text
    """
    return text.encode("utf-8").decode("utf-8")


def has_format_anomalies(fulltext: str, html_text: str) -> bool:
    """
    Checks if the text contains formatting anomalies or weird characters that could compromise
    the readability of the article
    """

    if len(html_text) < 0.95 * len(fulltext):
        logger.info("Anomaly: html_text is shorter than fulltext")
        return True

    patterns = [
        # text doesn't start with a capital letter, quotation marks or a digit
        r"^(?!(iPhone|iPad|iMac|iPod|eBay))[^A-ZÀ-Ý\"“«\d]+",
        r"[^\.!\?\"”»…\)]$",  # text doesn't end with a punctuation mark
        # r"\b[a-z]+[A-Z][a-z]+\b",  # merged words
    ]

    for pattern in patterns:
        if match := re.search(pattern, fulltext):
            logger.info(f"Found anomaly in fulltext: {match.group()}")
            return True

    html_patterns = [
        # html_text contains asterisks (bug from trafilatura when parsing formatted text)
        r"\*",
    ]

    for pattern in html_patterns:
        if match := re.search(pattern, html_text):
            logger.info(f"Found anomaly in html_text: {match.group()}")
            return True

    return False

def clean_start_of_article(text: str, rm_uppercase: bool, rm_first_hyphen: bool) -> str:
    # Some newspapers repeat the first line (not relevant to the text of the article) twice. In case they are removed
    lines = text.split('\n')
    if len(lines) > 1 and lines[0] == lines[1]:
        text = '\n'.join(lines[2:])

    start_of_article_multiline = [r'Se hai scelto di non accettare i cookie di profilazione e tracciamento, puoi aderire all’abbonamento "Consentless" a un costo molto accessibile, oppure scegliere un altro abbonamento per accedere ad ANSA.it.',
                                  r"Ti invitiamo a leggere le Condizioni Generali di Servizio, la Cookie Policy e l'Informativa Privacy.",
                                  r"In aggiornamento",
                                  r"Simply sign up .* -- delivered directly to your inbox.",
                                  r"^#*\s*\*?(di).*?(\.|\n)"]

    # list containing the sentences to cut the text from (present ONLY at the beginning of the article)
    # r"^.*\b[a-z]+(?=[A-Z][a-z]+\b)"
    start_of_article = [r"^.*Abbonati.*", r"^[^\w\"«“\d\*\#]+", r"^[A-Z]+\."]

    if rm_uppercase:
        start_of_article.append(r"^\**[A-Z ]{5,}\b\**")

    for pattern in start_of_article_multiline:
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)

    for pattern in start_of_article:
        if match := re.match(pattern, text):
            text = text[match.end():]


    # Remove up to the first hyphen, if there is not a second one in the same sentence
    if rm_first_hyphen:
        if match := re.match(r"^([^\.\-\–\—]*(?<!(\"|”|»))(\s|\*)(-|–|—))[^\-\–\—]*?(\.|\n)", text):
            if len(match.group(1).split()) < 8:
                text = text[match.end(1):]

    return text

def clean_end_of_article(text: str) -> str:
    # list containing the sentences from which the text must be cut (present only at the end of the article)
    end_of_article = [r'\bcondividi\b', "continua a leggere", r"#*\s*leggi\s+anche", "leggi l'articolo completo",
                      r"\(?riproduzione riservata\)?", r"riproduzione riservata", "potrebbe interessarti anche", "abbonati per", "cronaca",
                      "le posizioni espresse in questo articolo", r"loading\.\.\.", r"-\s*Argomenti",
                      r"-\s*Altri Mondi", "per altri contenuti iscriviti", "la tua opinione è importante",
                      "Questo sito contribuisce all’audience", r"(#+\\s*)?brand connect", "Gentile lettore",
                      "Iscriviti alle newsletter", r"se vuoi iscriverti", r"[^/w+]Fin dalla sua nascita", 
                      r"\b[A-Za-z0-9._%+-]+@corriere\.it\b", r"(?<!('|’))ultima ora",
                      r'\d{1,2} [a-zA-Z]+ \d{4} \(modifica il \d{1,2} [a-zA-Z]+ \d{4} \| \d{1,2}:\d{1,2}\)',
                      "#{0,4} La newsletter diario politico", "#+ dai blog", r"\*{0,4}Questo articolo contribuisce",
                      "ogni venerdì, nella tua casella di posta elettronica", "consigli24:",
                      r"- dal lunedì al venerdì dalle ore", "i commenti dei lettori",
                      r"\-\s+\*\*Leggi qui\*\*il GdB in edicola oggi|\-\s+Leggi qui il GdB in edicola oggi",
                      r"\s*#+\s*lascia un commento", r'\badv\b', r'il più letto\n',
                      r'- dal\n\*\*lunedì\*\*.*?\*\*venerdì\*\*dalle ore| - dal\nlunedì al venerdì dalle ore',
                      r'\*{1,2}?\s*(questo articolo)[^\n\.]*(è pubblicato)',
                      r'lavoce è di tutti', 'ogni venerdì, nella tua casella di posta elettronica,',
                      r"- \w+ \|", r"- \d+ min", r"\s*(-|–|—)\s*$", r"(?<=\.\s)(-|–|—).*\.$",
                      r"\([^\)]*foto.*?\)$"]

    for pattern in end_of_article:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            text = text[:match.start()]

    return text

def replace_garbage(text: str, rm_ansa: bool = False) -> str:
    # list containing the sentences to be replaced with empty strings
    # (present at the beginning and in the middle of the article)
    to_replace = [r"\d+.+di lettura", r"#{0,4} Leggi anche\n(.*?\n)*?(?=\#+(?![\#]))",
                  r'(?<!\S)- a\b', r'g\+', r'\(©\)|©',
                  'contenuto riservato agli abbonati',  r'\n—', r'\(facebook\)',
                  r'sei già registrato / abbonato? accedi', r'- a\n- a', 'video su questo argomento',
                  '(?:\\*\\*Ascolta ora:\\*\\*|Ascolta ora:).*', r'(\*Mail: [^\*]+\*|Mail: [^\*]+)',
                  'articolo originariamente']

    if rm_ansa:
        patterns = ["In evidenza\n", "Extra\n", "LIVE\n"]
        to_replace.extend(patterns)

        # rm hours
        text = re.sub(r"^\d{2}:\d{2}", "", text, flags=re.MULTILINE)

    # The patterns in the 'to_replace' list are searched for and possibly replaced only once (count = 1),
    # this to avoid unintentional modification of the text
    for pattern in to_replace:
        text = re.sub(
                pattern, '', text, count=1, flags=re.IGNORECASE)

    return text

def clean_title(text: str, article: tr.bare_extraction) -> str:
    # create a list of patterns that search for every author who has contributed to the article
    # authors = [re.compile(r"(\*{0,3}?di\s*)?(\*{0,3}\s*)" + re.escape(author) + r"(\s*\*{0,3})?", flags=re.IGNORECASE)
    #            for author in get_authors_from_page_source(meta_authors=article['author'], for_scraping=True)]

    # creates a pattern for searching the title within the text of the article
    patterns = []
    if article['title']:
        if "-" in article['title']:
            patterns.append(r"^#{1,3}\s*" + re.escape(article['title'].split('-')[0].strip()))
        patterns.append(r"^#{1,3}\s*" + re.escape(article['title']))

    # check if the title or the authors are present in the article (usually the author is present in the following way:
    # "(...) of <Author Name> <Author Surname>")

    # if an author or title is present within the text, they are removed
    # for pattern in [*authors, title]:
    for pattern in patterns:
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)

    return text

def clean_hanging_lines(text: str) -> str:
    """
    remove every sentence that ends without punctuation and is not a heading
    (html_text keeps the headings, while fulltext does not)
    """
    return re.sub(r"^(?!#).*(?<!(\.|\?|:)\*{2})(?<!(\.|\?|:)\s\*{2})(?<!(\.|\?|:)\*{1})(?<!(\.|\?|:)\s\*{1})(?<!\.\")((?<=[\'\’\"“\w\d\+ ])|(?<=\*{2})|(?<=\*{1}))(\n|$)", "", text, flags=re.MULTILINE)

def clean_markdown(text: str) -> str:
    # all newlines before asterisks are replaced with a space, if they are not headings
    text = re.sub(r"(^[^#].*)(\n)(?=\*)", r"\1 ", text.strip(), flags=re.MULTILINE)

    # remove lines that contain only hashtags
    text = re.sub(r"^#+$", "", text, flags=re.MULTILINE)

    return text

def fix_text(article: tr.bare_extraction, rm_uppercase: bool = False, rm_hanging_lines: bool = False,
             rm_first_hyphen: bool = False, rm_ansa: bool = False, include_formatting: bool = False) -> str:
    """
    Cleans up the text of the article by filtering through appropriate regular expressions

    Parameters:
    ----------
    article_data: trafilatura.bare_extraction
        dictionary containing all data about the article, such as title, site name, text, author, tags, category...

    Returns:
    -------
    str
        the cleaned text
    """

    text = article['text']

    # logger.info(f"Before cleaning:\n{text}\n\n")

    text = clean_start_of_article(text, rm_uppercase, rm_first_hyphen)
    text = clean_end_of_article(text)
    text = clean_markdown(text)

    if rm_hanging_lines:
        text = clean_hanging_lines(text)

    text = clean_title(text, article)
    text = replace_garbage(text, rm_ansa)

    # a second pass is sometimes needed after the replacements
    text = clean_start_of_article(text, rm_uppercase, rm_first_hyphen)
    text = clean_end_of_article(text)
    if rm_hanging_lines:
        text = clean_hanging_lines(text)


    # all newlines and additional spaces are removed from the text
    text = re.sub(r'\n{2,}', '\n', text.strip())

    # remove titles at the end of the article
    lines = text.split('\n')
    while lines[-1].startswith("#"):
        lines = lines[:-1]

    text = '\n'.join(map(lambda x: x.strip("\n"), lines))

    if not include_formatting:
        # remove all newlines from text
        text = text.replace("\n", " ")

    return text

def fix_text_from_trafilatura(article: tr.bare_extraction, 
                              include_formatting: bool = False) -> str:
    return fix_text(article, rm_uppercase=True, rm_hanging_lines=True, rm_first_hyphen=True, include_formatting=include_formatting)

def scrape_fulltext_with_trafilatura(page_source: str,
                                     include_images: bool = False,
                                     include_formatting: bool = False,
                                     include_comments: bool = False,
                                     include_tables: bool = False) -> str:
    """
    This is a bridge function that serves to discriminate the operations to be done based
    on how you want the text to be returned (formatted or not)

    Parameters
    ----------

    page_source: str
        html source of the article

    include_images: bool, default False
        whether the text should include images (Markdown style) or not

    include_formatting: bool, default False
        whether the text should preserve Markdown formatting or not

    include_comments: bool, default False
        whether the text should include a comments section or not

    include_tables: bool, default False
        whether the text should include tabular data  or not


    Returns:
    -------
    str
        the cleaned text
    """
    page_source = page_source.replace(
        "<em><strong>", "<em>").replace("</strong></em>", "</em>")
    page_source = page_source.replace(
        "<strong><em>", "<em>").replace("</em></strong>", "</em>")
    article = tr.bare_extraction(
        page_source,
        include_images=include_images,
        include_formatting=include_formatting,
        include_comments=include_comments,
        include_tables=include_tables,
    )

    return fix_text_from_trafilatura(article=article, include_formatting=include_formatting)



def fix_markdown(text: str):
    """
    Fixes the markdown errors from the input text
    """

    # <em> text with incorrect spacing
    em_pattern = r'(\w?)\*(\s*)(.*?)(\s*)\*(\w?)'

    # Replace <em> text with correct spacing
    fixed_text = re.sub(em_pattern, lambda match:
                        (match.group(1) + " *" if match.group(1) else "*") + match.group(3) +
                        ("* " + match.group(5) if match.group(5) else "*"), text)

    # <strong> text with incorrect spacing
    strong_pattern = r'(\w?)\*\*(\s*)(.*?)(\s*)\*\*(\w?)'

    # Replace <strong> text with correct spacing
    fixed_text = re.sub(strong_pattern, lambda match:
                        (match.group(1) + " **" if match.group(1) else "**") + match.group(3) +
                        ("** " + match.group(5) if match.group(5) else "**"), fixed_text)

    return fixed_text


def fix_html(text: str):
    # replace all newlines not preceded by a tag with <br>
    text = re.sub(r"(?<!\>)\n", "<br>", text)

    # if </p> is preceded by one or more '*', remove them
    text = re.sub(r"\s*\*+\s*<\/p>", "</p>", text)

    # turn all headings into <h3> for consistency
    text = re.sub(r"<h\d>", "<h3>", text)
    text = re.sub(r"</h\d>", "</h3>", text)

    return text


def get_fulltext_from_page_source(page_source: str = "",
                                  include_images=False,
                                  include_formatting=False,
                                  include_comments=False,
                                  include_tables=False) -> str:
    '''
    Given the HTML source code of an article webpage, uses trafilatura to retrieve the page fulltext
    Returns the fulltext as a string.
    '''
    fulltext = str()
    logger.info(
        "Ingesting fulltext with Trafilatura from article webpage source.")

    try:
        if page_source:
            fulltext = scrape_fulltext_with_trafilatura(page_source,
                                                        include_images,
                                                        include_formatting,
                                                        include_comments,
                                                        include_tables)

            if include_formatting:
                md_text = html.escape(fix_markdown(fulltext))
                fulltext = fix_html(markdown.markdown(md_text))
    except Exception as e:
        logger.error(
            "Exception when retrieving article fulltext with trafilatura: %s" %
            e)

    if not fulltext:
        logger.info("...unable to retrieve fulltext from webpage source.")
    else:
        logger.info("...fulltext scraped, begins with: %s..." %
                    fulltext[:80])
    return fulltext

def get_page_source_with_trafilatura(url: str) -> str:
    '''
    retrieves the HTML source of the page. Returns the page source as string.
    '''
    page_source = ""
    try:
        if html_source := tr.fetch_url(url, decode=False):
            if (encoding := charset_detect(html_source.data)['encoding']) == "windows-1250":
                encoding = "iso-8859-1"

            page_source = html_source.data.decode(encoding)

        if page_source is None or page_source == "" or not isinstance(page_source, str):
            raise Exception(
                "trafilatura returned an empty or non-string page source")

    except Exception as exc:
        logger.error(
            "[NewspaperScraper] Exception `%s` occurred when getting html source from trafilatura for url: %s" %
            (exc, url))

    return page_source

def scrape_article(article_url: str) -> str:
    page_source = get_page_source_with_trafilatura(article_url)
    return get_fulltext_from_page_source(page_source=page_source)


if __name__ == "__main__":
    url = "https://www.ilgiornale.it/news/personaggi/boicottaggio-internazionale-minaccia-codacons-e-l-abbandono-2264291.html"

    print(f"Article fulltext:\n{scrape_article(url)}")
