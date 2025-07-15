# OCR & Image Handling Guide

## Overview

Many municipal PDFs contain scanned documents, images with text, diagrams, and mixed content. This guide details how to handle these cases for complete text extraction.

## Current Limitations

1. **No OCR**: Scanned pages return empty text
2. **No Image Processing**: Figures and diagrams ignored
3. **No Table Detection**: Complex layouts become jumbled
4. **No Language Detection**: Assumes English only

## OCR Implementation Plan

### 1. Detection Strategy

```python
class PageTypeDetector:
    """Detect if a page needs OCR"""
    
    @staticmethod
    def needs_ocr(page):
        # Extract text normally
        text = page.get_text()
        
        # Heuristics for OCR detection
        indicators = {
            'text_length': len(text.strip()),
            'word_count': len(text.split()),
            'char_density': len(text) / (page.rect.width * page.rect.height),
            'has_images': len(page.get_images()) > 0
        }
        
        # Decision logic
        if indicators['text_length'] < 100 and indicators['has_images']:
            return True  # Likely scanned page
        
        if indicators['char_density'] < 0.001:
            return True  # Too little text for size
            
        return False
```

### 2. OCR Pipeline

```python
class OCRProcessor:
    """Comprehensive OCR processing"""
    
    def __init__(self):
        # Multiple OCR engines for fallback
        self.engines = {
            'tesseract': TesseractOCR(),
            'easyocr': EasyOCR() if gpu_available else None,
            'paddleocr': PaddleOCR() if advanced_ocr else None
        }
        
    def process_page(self, page, dpi=300):
        # Convert to high-res image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess image
        img = self.preprocess(img)
        
        # Try OCR engines in order
        for engine_name, engine in self.engines.items():
            if engine:
                try:
                    result = engine.extract(img)
                    if result.confidence > 0.8:
                        return result.text
                except Exception as e:
                    logger.warning(f"{engine_name} failed: {e}")
                    
        return ""  # Fallback if all fail
        
    def preprocess(self, img):
        """Improve image quality for OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Deskew if needed
        angle = self.detect_skew(thresh)
        if abs(angle) > 0.5:
            thresh = self.rotate_image(thresh, angle)
            
        return Image.fromarray(thresh)
```

### 3. Language Support

```python
class MultilingualOCR:
    """Handle multiple languages in municipal docs"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.ocr_languages = {
            'en': 'eng',
            'es': 'spa', 
            'fr': 'fra',
            'zh': 'chi_sim'
        }
        
    def detect_and_extract(self, img):
        # Detect language from visible text or metadata
        lang = self.language_detector.detect(img)
        
        # Configure OCR for detected language
        tesseract_lang = self.ocr_languages.get(lang, 'eng')
        
        return pytesseract.image_to_string(
            img, 
            lang=tesseract_lang,
            config='--psm 3 --oem 3'  # Page segmentation mode
        )
```

## Image & Figure Handling

### 1. Image Extraction

```python
class ImageExtractor:
    """Extract and process images from PDFs"""
    
    def extract_images(self, page):
        images = []
        
        # Get all images on page
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            # Extract image
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            if pix.n - pix.alpha > 3:  # Convert CMYK to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
                
            img_data = {
                'index': img_index,
                'page': page.number,
                'pixmap': pix,
                'bbox': page.get_image_bbox(img),
                'size': (pix.width, pix.height)
            }
            
            images.append(img_data)
            
        return images
```

### 2. Image Analysis

```python
class ImageAnalyzer:
    """Analyze images for text and content"""
    
    def __init__(self):
        self.text_detector = TextInImageDetector()
        self.caption_generator = ImageCaptioner()
        
    def analyze(self, image_data):
        img = self.pixmap_to_pil(image_data['pixmap'])
        
        analysis = {
            'has_text': False,
            'text_content': '',
            'image_type': 'unknown',
            'caption': ''
        }
        
        # Check for text in image
        if self.text_detector.detect(img):
            analysis['has_text'] = True
            analysis['text_content'] = self.extract_text(img)
            
        # Classify image type
        analysis['image_type'] = self.classify_image(img)
        
        # Generate caption for non-text images
        if analysis['image_type'] in ['diagram', 'photo', 'chart']:
            analysis['caption'] = self.caption_generator.generate(img)
            
        return analysis
```

### 3. Diagram & Chart Processing

```python
class DiagramProcessor:
    """Special handling for diagrams and charts"""
    
    def process_chart(self, img):
        """Extract data from charts"""
        # Detect chart type
        chart_type = self.detect_chart_type(img)
        
        if chart_type == 'bar':
            return self.extract_bar_chart_data(img)
        elif chart_type == 'pie':
            return self.extract_pie_chart_data(img)
        elif chart_type == 'line':
            return self.extract_line_chart_data(img)
            
        return None
        
    def process_diagram(self, img):
        """Extract text and relationships from diagrams"""
        # Detect text regions
        text_regions = self.detect_text_regions(img)
        
        # Extract text from each region
        texts = []
        for region in text_regions:
            text = self.ocr_engine.extract(region)
            texts.append({
                'text': text,
                'position': region.bbox,
                'confidence': region.confidence
            })
            
        # Detect connections/arrows
        connections = self.detect_connections(img)
        
        return {
            'texts': texts,
            'connections': connections,
            'type': 'diagram'
        }
```

## Table Extraction

### 1. Table Detection

```python
class TableDetector:
    """Detect and extract tables from PDFs"""
    
    def detect_tables(self, page):
        # Method 1: Use PDF structure
        tables = page.find_tables()
        
        if not tables:
            # Method 2: Visual detection
            img = self.page_to_image(page)
            tables = self.detect_tables_visual(img)
            
        return tables
        
    def detect_tables_visual(self, img):
        """Use computer vision to find tables"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal = self.detect_lines(gray, 'horizontal')
        vertical = self.detect_lines(gray, 'vertical')
        
        # Find intersections
        intersections = cv2.bitwise_and(horizontal, vertical)
        
        # Find table regions
        contours, _ = cv2.findContours(
            intersections, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum table size
                x, y, w, h = cv2.boundingRect(contour)
                tables.append({
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.9
                })
                
        return tables
```

### 2. Table Data Extraction

```python
class TableExtractor:
    """Extract structured data from tables"""
    
    def extract_table_data(self, table_region, page):
        # Try camelot for PDF tables
        try:
            tables = camelot.read_pdf(
                page.parent.name,
                pages=str(page.number + 1),
                flavor='lattice',  # or 'stream'
                table_areas=[table_region]
            )
            
            if tables:
                return tables[0].df.to_dict('records')
        except:
            pass
            
        # Fallback to tabula
        try:
            tables = tabula.read_pdf(
                page.parent.name,
                pages=page.number + 1,
                area=table_region,
                pandas_options={'header': None}
            )
            
            if tables:
                return tables[0].to_dict('records')
        except:
            pass
            
        # Last resort: OCR
        return self.ocr_table(table_region, page)
```

## Integration with MuniRAG

### 1. Enhanced PDF Processor

```python
class EnhancedPDFProcessor:
    """PDF processor with full OCR and image support"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.ocr_processor = OCRProcessor()
        self.image_analyzer = ImageAnalyzer()
        self.table_extractor = TableExtractor()
        
    def process_page(self, page):
        results = {
            'text': '',
            'images': [],
            'tables': [],
            'metadata': {}
        }
        
        # Extract regular text
        results['text'] = page.get_text()
        
        # Check if OCR needed
        if PageTypeDetector.needs_ocr(page):
            ocr_text = self.ocr_processor.process_page(page)
            results['text'] += '\n\n[OCR Content]\n' + ocr_text
            
        # Extract images
        images = self.image_analyzer.extract_images(page)
        for img in images:
            analysis = self.image_analyzer.analyze(img)
            if analysis['has_text']:
                results['text'] += f"\n\n[Image {img['index']} Text]\n{analysis['text_content']}"
            elif analysis['caption']:
                results['text'] += f"\n\n[Image {img['index']}]: {analysis['caption']}"
                
        # Extract tables
        tables = self.table_extractor.detect_tables(page)
        for table in tables:
            data = self.table_extractor.extract_table_data(table, page)
            # Convert table to markdown for text embedding
            table_text = self.table_to_markdown(data)
            results['text'] += f"\n\n[Table]\n{table_text}"
            
        return results
```

### 2. Configuration

```yaml
# OCR Settings
OCR_ENABLE: true
OCR_ENGINE: tesseract  # tesseract|easyocr|paddleocr
OCR_LANGUAGES: [eng, spa]
OCR_DPI: 300
OCR_CONFIDENCE_THRESHOLD: 0.8
OCR_PREPROCESS: true

# Image Settings  
IMAGE_EXTRACT: true
IMAGE_CAPTION: true
IMAGE_MAX_SIZE_MB: 10
IMAGE_FORMATS: [png, jpg, jpeg, tiff]

# Table Settings
TABLE_EXTRACT: true
TABLE_ENGINE: camelot  # camelot|tabula|ocr
TABLE_MIN_ROWS: 2
TABLE_TO_MARKDOWN: true
```

### 3. Performance Considerations

```python
class OCROptimizer:
    """Optimize OCR performance"""
    
    def __init__(self):
        self.cache = {}
        self.gpu_available = torch.cuda.is_available()
        
    def batch_ocr(self, images):
        """Process multiple images in batch"""
        if self.gpu_available and len(images) > 5:
            # Use GPU-accelerated OCR
            return self.gpu_batch_ocr(images)
        else:
            # Parallel CPU processing
            with ProcessPoolExecutor() as executor:
                results = executor.map(self.process_single, images)
            return list(results)
            
    def cache_results(self, image_hash, result):
        """Cache OCR results to avoid reprocessing"""
        self.cache[image_hash] = {
            'text': result,
            'timestamp': time.time()
        }
```

## Testing Strategy

### 1. Test Cases
- Pure text PDF (baseline)
- Scanned PDF (100% images)
- Mixed PDF (text + scanned pages)
- PDF with diagrams and charts
- PDF with complex tables
- Multi-language PDF
- Low-quality scanned PDF

### 2. Metrics
- OCR accuracy (vs ground truth)
- Processing time per page
- Memory usage
- Text extraction completeness
- Table structure preservation

### 3. Validation
```python
def validate_ocr_output(original_pdf, extracted_text):
    """Validate OCR quality"""
    metrics = {
        'char_count': len(extracted_text),
        'word_count': len(extracted_text.split()),
        'confidence': calculate_confidence(extracted_text),
        'language': detect_language(extracted_text),
        'completeness': estimate_completeness(original_pdf, extracted_text)
    }
    
    return metrics
```

## Deployment Checklist

- [ ] Install OCR dependencies (Tesseract, OpenCV)
- [ ] Configure language packs
- [ ] Set up image preprocessing pipeline
- [ ] Add table extraction libraries
- [ ] Configure GPU for accelerated OCR (optional)
- [ ] Set up caching for processed images
- [ ] Add monitoring for OCR performance
- [ ] Create fallback strategies
- [ ] Document OCR limitations
- [ ] Test with real municipal PDFs

This comprehensive approach ensures no content is missed during PDF processing, significantly improving the accuracy and completeness of the RAG system.