# Python Imports
import os
import sys
import traceback
from pathlib import Path
from pprint import pprint
from collections import OrderedDict

# Library Imports
from pypdf import PdfReader

# Custom Imports
from library.utils.log import applog
from appconfig import properties

# Gloabal Variable/Settings

class Size(int):

    _KB = 1024
    _suffixes = 'B', 'KB', 'MB', 'GB', 'PB'

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.bytes = self.B = int(self)
        self.kilobytes = self.KB = self / self._KB**1
        self.megabytes = self.MB = self / self._KB**2
        self.gigabytes = self.GB = self / self._KB**3
        self.petabytes = self.PB = self / self._KB**4
        *suffixes, last = self._suffixes
        suffix = next((
            suffix
            for suffix in suffixes
            if 1 < getattr(self, suffix) < self._KB
        ), last)
        self.readable = suffix, getattr(self, suffix)

        super().__init__()

    def __str__(self):
        return self.__format__('.2f')

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, super().__repr__())

    def __format__(self, format_spec):
        suffix, val = self.readable
        return '{val:{fmt}} {suf}'.format(val=val, fmt=format_spec, suf=suffix)

    def __sub__(self, other):
        return self.__class__(super().__sub__(other))

    def __add__(self, other):
        return self.__class__(super().__add__(other))
    
    def __mul__(self, other):
        return self.__class__(super().__mul__(other))

    def __rsub__(self, other):
        return self.__class__(super().__sub__(other))

    def __radd__(self, other):
        return self.__class__(super().__add__(other))
    
    def __rmul__(self, other):
        return self.__class__(super().__rmul__(other))   

class IsFolder:

    def __init__(self, location):
        self.location = location
        if Path(f'{self.location}').is_dir():
            pass
        else:
            raise FileNotFoundError(os.path.basename(location)+' folder do not exist in the location - '+os.path.dirname(location))

def get_folder_size(folder):
    return Size(sum(file.stat().st_size for file in Path(folder).rglob('*')))

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.4f %s" % (num, x)
        num /= 1024.0

def get_dir_size(path):
    
    size = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            size +=  os.path.getsize(os.path.join(root, file))
   
    return size

def analyze_text(string: str) -> bool:
    """
    Validate if a string is contains atleast some characters

    Args:
        string (str): Input string to be analyzed

    Returns:
        bool: True if string is null or whitespace or new line characters only, False otherwise.
    """
    
    if (string is None) or (string == "") or (string.isspace()):
        return True
    
    return False

def analyze_pdf(file_absolute_path: str) -> dict:

    pdf_metadata   = OrderedDict()
    pages_metadata = []
    pdf_pypdf      = PdfReader(file_absolute_path)
    
    pdf_metadata["encrypted"] = pdf_pypdf.is_encrypted

    if not pdf_pypdf.is_encrypted:
        pages_count = len(pdf_pypdf.pages)
        
        for pageno in range(0, pages_count):
            page_metadata = OrderedDict()
            page_data = pdf_pypdf.pages[pageno]
            page_text = page_data.extract_text()
            page_type = "scanned" if analyze_text(page_text) else "searchable"
            page_metadata["page_number"] = pageno+1
            page_metadata["page_type"]   = page_type
            pages_metadata.append(page_metadata)
        
        pdf_metadata["pages_count"]     = pages_count
        pdf_metadata["pages_metadata"]  = pages_metadata
    else:
        pdf_metadata["pages_count"]     = 0
        pdf_metadata["pages_metadata"]  = pages_metadata

    return pdf_metadata

def analyze_file(file_absolute_path: str) -> dict:
    
    file_metadata                                           = OrderedDict()
    file_name, file_extension                               = os.path.splitext(file_absolute_path)
    file_metadata["file_absolute_path_with_extension"]      = file_absolute_path
    file_metadata["file_absolute_path_without_extension"]   = file_name
    file_metadata["file_folder"]                            = os.path.dirname(file_absolute_path)
    file_metadata["file_name_without_extension"]            = os.path.basename(file_name)
    file_metadata["file_name_with_extension"]               = os.path.basename(file_absolute_path)
    file_metadata["file_extension"]                         = file_extension
    file_metadata["file_size"]                              = convert_bytes(os.path.getsize(file_absolute_path))
    
    if file_extension.lower() in properties["files"]["pdf_extensions"]:
        file_metadata["pdf_metadata"] = analyze_pdf(file_absolute_path)

    return file_metadata

def analyze_path(path: str) -> dict:
    
    try:
        IsFolder(path)
        path_metadata   = OrderedDict()
        dirs_size       = OrderedDict()
        files_metadata  = OrderedDict()
        all_files       = []
        for root, dirs, files in os.walk(path):
            
            all_files.extend(files)
            
            for file in files:
                file_metadata = analyze_file(os.path.join(root, file))
                files_metadata[file] = file_metadata
            
            if dirs:
                for dir in dirs:
                    dirs_size[dir] = convert_bytes(get_dir_size(os.path.join(root, dir)))
        
        path_metadata["root_size"]      = convert_bytes(get_dir_size(path))
        path_metadata["dirs_size"]      = dirs_size
        path_metadata["files"]          = all_files
        path_metadata["files_metadata"] = files_metadata
    except FileNotFoundError:
        applog.catcherror( message="Path does not exists - "+path, task="exception")
        exit()   

    return path_metadata