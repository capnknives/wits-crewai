def __init__(self, quartermaster=None):
        super().__init__()
        self.quartermaster = quartermaster  # For reading files
        self.output_dir = "documents"
        # Create output directory if quartermaster is provided
        if self.quartermaster:
            try:
                # This will create the directory if it doesn't exist
                self.quartermaster.list_files(self.output_dir)
            except:
                # Make sure the directory exists
                os.makedirs(os.path.join(self.quartermaster.output_dir, self.output_dir), exist_ok=True)
    
    def _get_timestamp(self):
        """Get a timestamp string for file naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _convert_markdown_to_reportlab(self, markdown_text):
        """Convert Markdown text to a list of reportlab elements"""
        styles = getSampleStyleSheet()
        
        # Create custom styles for different heading levels and other elements
        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12
        )
        heading2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=10
        )
        heading3_style = ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8
        )
        code_style = ParagraphStyle(
            'Code',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=8,
            backColor=colors.lightgrey
        )
        
        # Split the markdown text into lines
        lines = markdown_text.split('\n')
        
        # Process the lines to handle Markdown formatting
        elements = []
        i = 0
        in_code_block = False
        code_block_content = []
        in_list = False
        list_items = []
        
        while i < len(lines):
            line = lines[i]
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    code_text = '\n'.join(code_block_content)
                    elements.append(Paragraph(code_text, code_style))
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                i += 1
                continue
            
            if in_code_block:
                code_block_content.append(line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
                i += 1
                continue
            
            # Handle headings
            if line.strip().startswith('# '):
                elements.append(Paragraph(line.strip()[2:], heading1_style))
            elif line.strip().startswith('## '):
                elements.append(Paragraph(line.strip()[3:], heading2_style))
            elif line.strip().startswith('### '):
                elements.append(Paragraph(line.strip()[4:], heading3_style))
            # Handle unordered lists
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    in_list = True
                    list_items = []
                list_items.append(line.strip()[2:])
            # Handle ordered lists (basic support)
            elif re.match(r'^\d+\.\s', line.strip()):
                if not in_list:
                    in_list = True
                    list_items = []
                list_items.append(re.sub(r'^\d+\.\s', '', line.strip()))
            # Empty lines or other content
            else:
                # If we were in a list, process it now
                if in_list and list_items:
                    # Create a bullet list
                    bullet_text = '<bullet>\u2022</bullet>'
                    for item in list_items:
                        elements.append(Paragraph(f"{bullet_text} {item}", styles['Normal']))
                    list_items = []
                    in_list = False
                
                # Process normal paragraph text
                if line.strip():
                    # Handle basic inline formatting
                    # Bold
                    line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                    # Italic
                    line = re.sub(r'\*(.+?)\*', r'<i>\1</i>', line)
                    # Code
                    line = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', line)
                    
                    elements.append(Paragraph(line, styles['Normal']))
                else:
                    elements.append(Spacer(1, 0.1 * inch))
            
            i += 1
        
        # Process any remaining list items
        if in_list and list_items:
            bullet_text = '<bullet>\u2022</bullet>'
            for item in list_items:
                elements.append(Paragraph(f"{bullet_text} {item}", styles['Normal']))
        
        return elements
    
    def _process_image(self, image_info, quartermaster=None):
        """Process image information to get a reportlab Image object"""
        img_data = None
        
        # Case 1: Image provided as base64 data
        if 'data' in image_info:
            try:
                # Decode base64 to binary
                img_data = base64.b64decode(image_info['data'])
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(img_data)
                    return Image(temp_file.name, width=4*inch, height=3*inch)
            except Exception as e:
                raise ValueError(f"Error processing base64 image: {str(e)}")
        
        # Case 2: Image provided as a path
        elif 'path' in image_info and quartermaster:
            try:
                img_path = image_info['path']
                # Try to read the file via quartermaster
                img_content = quartermaster.read_file(img_path)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_path)[1]) as temp_file:
                    temp_file.write(img_content.encode() if isinstance(img_content, str) else img_content)
                    return Image(temp_file.name, width=4*inch, height=3*inch)
            except Exception as e:
                raise ValueError(f"Error loading image from path '{img_path}': {str(e)}")
        
        # Case 3: Direct path to an image file
        elif 'path' in image_info:
            img_path = image_info['path']
            if os.path.exists(img_path):
                return Image(img_path, width=4*inch, height=3*inch)
            else:
                raise ValueError(f"Image file not found: {img_path}")
        
        else:
            raise ValueError("Image info must contain either 'data' (base64) or 'path' key")
    
    def _create_header_footer(self, canvas, doc, header_text, footer_text):
        """Add header and footer to PDF pages"""
        canvas.saveState()
        styles = getSampleStyleSheet()
        
        # Header
        if header_text:
            header_style = ParagraphStyle(
                'Header',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey
            )
            p = Paragraph(header_text, header_style)
            w, h = p.wrap(doc.width, doc.topMargin)
            p.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)
            
            # Add a line under the header
            canvas.setStrokeColor(colors.lightgrey)
            canvas.line(doc.leftMargin, doc.height + doc.topMargin - h - 5,
                       doc.leftMargin + doc.width, doc.height + doc.topMargin - h - 5)
        
        # Footer with page number
        if footer_text:
            footer = footer_text
            if "%(page)d" not in footer_text:
                footer += " | Page %(page)d" % {"page": doc.page}
            else:
                footer = footer_text % {"page": doc.page}
                
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=1  # Center
            )
            p = Paragraph(footer, footer_style)
            w, h = p.wrap(doc.width, doc.bottomMargin)
            p.drawOn(canvas, doc.leftMargin, h)
            
            # Add a line above the footer
            canvas.setStrokeColor(colors.lightgrey)
            canvas.line(doc.leftMargin, h + 5,
                       doc.leftMargin + doc.width, h + 5)
        
        canvas.restoreState()
    
    def execute(self, **kwargs) -> str:
        # Extract and validate parameters
        content = kwargs.get("content")
        title = kwargs.get("title", "Generated Document")
        output_path = kwargs.get("output_path", "")
        page_size = kwargs.get("page_size", "letter").lower()
        include_toc = kwargs.get("include_toc", False)
        header_text = kwargs.get("header_text", "")
        footer_text = kwargs.get("footer_text", "Page %(page)d")
        images = kwargs.get("images", [])
        
        # Validate required parameters
        if not content:
            raise ToolException("PdfGeneratorTool: 'content' parameter is required.")
        
        # Validate page size
        if page_size == "letter":
            page_dimensions = letter
        elif page_size == "a4":
            page_dimensions = A4
        else:
            raise ToolException("PdfGeneratorTool: 'page_size' must be either 'letter' or 'a4'.")
        
        try:
            # Determine output file path
            if not output_path:
                timestamp = self._get_timestamp()
                output_path = f"{self.output_dir}/document_{timestamp}.pdf"
            
            # Ensure the output path has .pdf extension
            if not output_path.lower().endswith('.pdf'):
                output_path += '.pdf'
            
            # If path is not absolute and quartermaster is available, save to output directory
            if not os.path.isabs(output_path) and self.quartermaster:
                full_path = os.path.join(self.quartermaster.output_dir, output_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
            else:
                # Create directories if needed
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                full_path = output_path
            
            # Create a PDF document
            def header_footer(canvas, doc):
                self._create_header_footer(canvas, doc, header_text, footer_text)
            
            doc = SimpleDocTemplate(
                full_path,
                pagesize=page_dimensions,
                title=title
            )
            
            # Process the content
            elements = []
            
            # Add title
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=36
            )
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 0.25 * inch))
            
            # Process the main content
            content_elements = self._convert_markdown_to_reportlab(content)
            elements.extend(content_elements)
            
            # Process images
            for img_info in images:
                try:
                    img = self._process_image(img_info, self.quartermaster)
                    elements.append(img)
                    elements.append(Spacer(1, 0.25 * inch))
                    
                    # Add caption if provided
                    if 'caption' in img_info:
                        caption_style = ParagraphStyle(
                            'Caption',
                            parent=styles['Normal'],
                            fontSize=10,
                            italic=True,
                            alignment=1  # Center
                        )
                        elements.append(Paragraph(img_info['caption'], caption_style))
                        elements.append(Spacer(1, 0.5 * inch))
                except Exception as e:
                    print(f"[PdfGeneratorTool] Error processing image: {e}")
                    # Add error message in the document
                    error_style = ParagraphStyle(
                        'Error',
                        parent=styles['Normal'],
                        textColor=colors.red
                    )
                    elements.append(Paragraph(f"Error including image: {str(e)}", error_style))
                    elements.append(Spacer(1, 0.25 * inch))
            
            # Build the PDF
            doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)
            
            return f"PDF document created successfully! Saved to: {output_path}"
        
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"[PdfGeneratorTool] Error: {trace}")
            raise ToolException(f"PdfGeneratorTool: Failed to create PDF document. Error: {str(e)}")
# agents/tools/pdf_generator_tool.py
from .base_tool import Tool, ToolException
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import os
import re
import markdown
import tempfile
from pathlib import Path
import base64
from io import BytesIO

class PdfGeneratorTool(Tool):
    name = "generate_pdf"
    description = ("Creates PDF documents from provided content. "
                   "The tool supports text formatting, tables, and images (if provided in base64 format). "
                   "It can generate reports, documentation, and other professional documents.")
    argument_schema = {
        "content": "str: The text content for the PDF document. Can be plain text or Markdown formatted.",
        "title": "str (optional): The title of the document",
        "output_path": "str (optional): Path where the PDF should be saved",
        "page_size": "str (optional): 'letter' or 'a4', defaults to 'letter'",
        "include_toc": "bool (optional): Whether to include a table of contents, defaults to False",
        "header_text": "str (optional): Text to appear in the document header",
        "footer_text": "str (optional): Text to appear in the document footer",
        "images": "list (optional): List of dictionaries with 'name' and 'data' (base64) or 'path' keys for images to include"
    }