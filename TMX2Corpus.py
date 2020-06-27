'''
Created on Dec 19, 2013
A script to convert TMXs into parallel corpuses for machine
translation (e.g. Moses: http://www.statmt.org/moses/) training.
Pass in either paths to TMX files, or directories containing TMX files.
The script will recursively traverse directories and process all TMXs.
To perform tokenization or to filter the output, use the convert() method
with subclasses of the Tokenizer or Filter objects.
This script is Jython 2.5-compatible, so you can use nice Java tokenizers
like those included in Lucene (http://lucene.apache.org/).
@author: aaron.madlon-kay
'''

from __future__ import with_statement
import sys
import os
import codecs
from xml.etree import ElementTree


class Tokenizer(object):
    def __init__(self, lang):
        self.lang = lang

    def tokenize(self, text):
        '''Override this to return tokenized text.'''
        return text


class Filter(object):
    def filter(self, bitext):
        '''Override this to return a bool indicating whether or not to keep the segment.'''
        return True


class Converter(object):
    def __init__(self):
        self.tokenizers = {}
        self.filters = []
        self.suppress_count = 0
        self.output_files = {}
        self.output_path = os.getcwd()
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        cleanup(self.output_files)

    def add_tokenizers(self, tokenizers):
        for tokenizer in tokenizers:
            self.tokenizers[tokenizer.lang] = tokenizer

    def add_filter(self, bitext_filter):
        if bitext_filter != None:
            self.filters.append(bitext_filter)

    def convert(self, files):
        self.suppress_count = 0
        try:
            for tmx in files:
                for bitext in extract_tmx(tmx):
                    self.__output(bitext)
            print('done')
        finally:
            print(f'Suppressed {self.suppress_count:d} pairs')

    def __output(self, bitext):
        for fltr in self.filters:
            if not fltr.filter(bitext):
                self.suppress_count += 1
                return

        for lang, text in bitext.items():
            tokenizer = self.tokenizers.get(lang, None)
            if tokenizer != None:
                bitext['tok.' + lang] = tokenizer.tokenize(text)

        for lang in bitext.keys():
            if lang not in self.output_files.keys():
                self.output_files[lang] = codecs.open(os.path.join(self.output_path, 'bitext.' + lang),
                                                      'w',
                                                      encoding='utf-8')
        for lang, text in bitext.items():
            out_file = self.output_files[lang]
            out_file.write(text)
            out_file.write('\n')


def get_files(path, ext):
    for root, dirs, files in os.walk(path):
        for a_file in files:
            if a_file.endswith(ext):
                yield os.path.join(root, a_file)


def extract_tmx(tmx):
    print('Extracting', os.path.basename(tmx))
    tree = ElementTree.parse(tmx)
    root = tree.getroot()
    for tu in root.getiterator('tu'):
        bitext = extract_tu(tu)
        if bitext != {}:
            yield bitext


def extract_tu(tu):
    bitext = {}
    for tuv in tu.findall('tuv'):
        lang, text = extract_tuv(tuv)
        if None not in (lang, text):
            bitext[lang] = text
    if len(bitext) != 2:
        print('TU had %d TUV(s). Skipping.' % len(bitext))
        print('\t' + ElementTree.tostring(tu))
        return {}
    return bitext


def extract_tuv(tuv):
    lang = tuv.attrib.get('lang', None)
    if lang == None:
        lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang', None)
    if lang == None:
        print('TUV missing lang. Skipping.')
        return None, None
    lang = normalize_lang(lang)
    segs = tuv.findall('seg')
    if len(segs) > 1:
        print('Multiple segs found in TUV. Skipping.')
        return None, None
    text = extract_seg(segs[0])
    if text == None:
        print('TUV missing seg. Skipping.')
        return None, None
    text = text.strip().replace('\n', '').replace('\r', '')
    if len(text) == 0:
        print('TUV had blank seg. Skipping.')
        return None, None
    return lang, text


def extract_seg(seg):
    buffer = [seg.text]
    for child in seg.getchildren():
        buffer.append(child.text)
        buffer.append(child.tail)
    return ''.join([piece for piece in buffer if piece != None])


def normalize_lang(lang):
    result = lang.lower()
    if len(result) > 2 and result[2] in ('-', '_'):
        result = result[:2]
    return result


def cleanup(output_files):
    for lang, out_file in output_files.items():
        out_file.close()


def convert(paths, tokenizers=[], bitext_filter=None):
    files = []
    for path in paths:
        if os.path.isdir(path):
            print('Queuing TMXs in ' + path)
            files.extend(get_files(path, '.tmx'))
        elif os.path.isfile(path) and path.endswith('.tmx'):
            files.append(path)
    if len(files) > 0:
        with Converter() as converter:
            converter.add_tokenizers(tokenizers)
            converter.add_filter(bitext_filter)
            converter.convert(files)
    else:
        print('Please specify input files or paths.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(convert(["/home/christian/Downloads/JRC-Acquis-v3.0"]))