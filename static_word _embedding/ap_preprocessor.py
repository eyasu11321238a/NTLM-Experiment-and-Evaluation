# AP Data
# Function to parse the TREC file
def AP_parse_trec_file(trec_file_path):
    doc_texts = {}
    current_doc_id = None
    current_text = []
    
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
    for encoding in encodings:
        try:
            with open(trec_file_path, 'r', encoding=encoding, errors='ignore') as file:
                for line in file:
                    if line.startswith('<DOCNO>'):
                        current_doc_id = line.strip().replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                    elif line.startswith('</TEXT>'):
                        if current_doc_id:
                            doc_texts[current_doc_id] = ' '.join(current_text)
                            current_doc_id = None
                            current_text = []
                    elif current_doc_id:
                        if not (line.startswith('<DOC>') or line.startswith('</DOC>') or line.startswith('<FILEID>') or
                                line.startswith('<FIRST>') or line.startswith('<SECOND>') or line.startswith('<HEAD>') or
                                line.startswith('</BYLINE>') or
                                line.startswith('<DATELINE>') or line.startswith('<TEXT>')):
                            current_text.append(line.strip())
            break
        except UnicodeDecodeError:
            continue  

    return doc_texts

