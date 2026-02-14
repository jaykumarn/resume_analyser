from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# Load pretrained NER model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    entities = ner_pipeline(text)
    result = {}
    for ent in entities:
        label = ent['entity_group']
        if label not in result:
            result[label] = []
        result[label].append(ent['word'])
    # Deduplicate
    for key in result:
        result[key] = list(set(result[key]))
    return result

def extract_resume_sections(text):
    """Extract Skills, Education, and Experience sections from resume text."""
    sections = {
        'skills': [],
        'education': [],
        'experience': []
    }
    
    section_headers = [
        'skills', 'technical skills', 'core competencies',
        'education', 'academic', 'qualification',
        'experience', 'work experience', 'employment', 'professional experience',
        'summary', 'objective', 'projects', 'certifications'
    ]
    
    pattern = r'(?i)^(' + '|'.join(re.escape(h) for h in section_headers) + r')[\s:]*$'
    lines = text.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        header_match = re.match(pattern, line_stripped)
        if header_match:
            if current_section and current_content:
                _store_section(sections, current_section, current_content)
            
            current_section = header_match.group(1).lower()
            current_content = []
        elif current_section:
            current_content.append(line_stripped)
    
    if current_section and current_content:
        _store_section(sections, current_section, current_content)
    
    return sections

def _store_section(sections, header, content):
    """Map header to appropriate section key and store content."""
    content_text = '\n'.join(content)
    
    if header in ['skills', 'technical skills', 'core competencies']:
        sections['skills'] = _parse_skills(content_text)
    elif header in ['education', 'academic', 'qualification']:
        sections['education'] = content
    elif header in ['experience', 'work experience', 'employment', 'professional experience']:
        sections['experience'] = content

def _parse_skills(text):
    """Parse skills from comma/newline separated text."""
    skills = re.split(r'[,\n]', text)
    return [s.strip() for s in skills if s.strip()]

def extract_contact_details(text):
    """Extract contact details (name, email, phone, LinkedIn) from resume text."""
    contact = {
        'name': None,
        'email': None,
        'phone': None,
        'linkedin': None
    }
    
    # Extract email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    if email_match:
        contact['email'] = email_match.group()
    
    # Extract phone number (various formats)
    phone_patterns = [
        r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\+?\d{1,3}[-.\s]?\d{10}',  # International
        r'\d{10}',  # Simple 10 digit
        r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # +1 234 567 8901
    ]
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group()
            break
    
    # Extract LinkedIn URL
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
    if linkedin_match:
        contact['linkedin'] = linkedin_match.group()
    
    # Extract name (usually first line or line with "Resume" removed)
    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # Remove common suffixes like "- Resume", "Resume", etc.
        name = re.sub(r'\s*[-–]\s*(Resume|CV).*$', '', first_line, flags=re.IGNORECASE)
        name = re.sub(r'\s*(Resume|CV)$', '', name, flags=re.IGNORECASE)
        name = name.strip()
        # Validate it looks like a name (2-4 words, no special chars except spaces)
        if name and re.match(r'^[A-Za-z]+(?:\s+[A-Za-z]+){0,3}$', name):
            contact['name'] = name
    
    return contact
