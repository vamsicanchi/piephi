import json
import pdfplumber
from pprint import pprint

def apply_pdfplumber(file, pdfplumber_config):

    with pdfplumber.open(file) as pdf:

        pdf_data                = {}
        pdf_data["metadata"]    = pdf.metadata
        pdf_data["pdf_pages"]   = pdf.pages
        for pdf_page in pdf.pages:
            page_no                     = pdf_page.page_number
            page_temp                   = {}
            page_temp["objects"]        = pdf_page.objects
            page_temp["chars"]          = pdf_page.chars 
            page_temp["lines"]          = pdf_page.lines
            page_temp["rects"]          = pdf_page.rects
            page_temp["curves"]         = pdf_page.curves
            page_temp["width"]          = pdf_page.width
            page_temp["height"]         = pdf_page.height
            page_temp["alltext_simple"] = pdf_page.extract_text(
                                                                x_tolerance = pdfplumber_config["extract_text_simple"]["x_tolerance"],
                                                                y_tolerance = pdfplumber_config["extract_text_simple"]["y_tolerance"]
                                                               )
            page_temp["alltext"]        = pdf_page.extract_text(
                                                                x_tolerance = pdfplumber_config["extract_text"]["x_tolerance"],
                                                                y_tolerance = pdfplumber_config["extract_text"]["y_tolerance"],
                                                                layout      = pdfplumber_config["extract_text"]["layout"],
                                                                x_density   = pdfplumber_config["extract_text"]["x_density"],
                                                                y_density   = pdfplumber_config["extract_text"]["y_density"]
                                                               )
            page_temp["words"]          = pdf_page.extract_words(
                                                                 x_tolerance          = pdfplumber_config["extract_words"]["x_tolerance"],
                                                                 y_tolerance          = pdfplumber_config["extract_words"]["y_tolerance"],
                                                                 keep_blank_chars     = pdfplumber_config["extract_words"]["keep_blank_chars"],
                                                                 use_text_flow        = pdfplumber_config["extract_words"]["use_text_flow"],
                                                                 horizontal_ltr       = pdfplumber_config["extract_words"]["horizontal_ltr"],
                                                                 vertical_ttb         = pdfplumber_config["extract_words"]["vertical_ttb"],
                                                                 extra_attrs          = pdfplumber_config["extract_words"]["extra_attrs"],
                                                                 split_at_punctuation = pdfplumber_config["extract_words"]["split_at_punctuation"],
                                                                 expand_ligatures     = pdfplumber_config["extract_words"]["expand_ligatures"]
                                                                )
            page_temp["alltext_lines"]  = pdf_page.extract_text_lines(
                                                                      layout       = pdfplumber_config["extract_text_lines"]["layout"],
                                                                      strip        = pdfplumber_config["extract_text_lines"]["strip"],
                                                                      return_chars = pdfplumber_config["extract_text_lines"]["return_chars"]
                                                                     )
            page_temp["alltext_search"] = pdf_page.search(
                                                          pattern       = pdfplumber_config["search"]["pattern"],
                                                          regex         = pdfplumber_config["search"]["regex"],
                                                          case          = pdfplumber_config["search"]["case"],
                                                          main_group    = pdfplumber_config["search"]["main_group"],
                                                          return_groups = pdfplumber_config["search"]["return_groups"],
                                                          return_chars  = pdfplumber_config["search"]["return_chars"],
                                                          layout        = pdfplumber_config["search"]["layout"]
                                                         )
            page_temp["alltext_dedupe"] = pdf_page.dedupe_chars(tolerance=pdfplumber_config["dedup_chars"]["tolerance"])

            page_temp["find_tables"]    = pdf_page.find_tables(table_settings=pdfplumber_config["find_tables"])
            page_temp["find_table"]     = pdf_page.find_table(table_settings=pdfplumber_config["find_table"])
            page_temp["extract_tables"] = pdf_page.extract_tables(table_settings=pdfplumber_config["extract_tables"])
            page_temp["extract_table"]  = pdf_page.extract_table(table_settings=pdfplumber_config["extract_table"])
            page_temp["debug_table"]    = pdf_page.debug_tablefinder(table_settings=pdfplumber_config["debug_tablefinder"])
            pdf_data[page_no]           = page_temp

    return pdf_data

pdfplumber_config_path = "D:\code\python\piephi\bin\configuration\properties.json"

with open(pdfplumber_config_path, 'r') as j:
    config = json.load(j)

ans = apply_pdfplumber("D:\datasets\pdf\invoice1.pdf")
pprint(ans)