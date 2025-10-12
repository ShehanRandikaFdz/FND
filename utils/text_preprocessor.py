"""
Text Preprocessor Module
Handles text preprocessing with step-by-step display matching training data preprocessing
"""

import re
from typing import Dict, List, Tuple


class TextPreprocessor:
    """
    Text preprocessing module that matches the training data preprocessing pipeline
    Provides step-by-step preprocessing for transparency and debugging
    """
    
    def __init__(self):
        """Initialize the text preprocessor"""
        self.preprocessing_steps = [
            "Original Text",
            "Convert to Lowercase", 
            "Remove URLs",
            "Remove Email Addresses",
            "Remove Phone Numbers",
            "Remove Special Characters",
            "Remove Extra Whitespace",
            "Final Processed Text"
        ]
    
    def preprocess(self, text: str, show_steps: bool = False) -> Dict:
        """
        Preprocess text matching the training data preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            show_steps: Whether to return step-by-step results
            
        Returns:
            Dictionary with original, processed text and optional steps
        """
        if not text:
            return {
                'original': '',
                'processed': '',
                'steps': []
            }
        
        steps = []
        current_text = text
        
        # Step 1: Store original
        if show_steps:
            steps.append(('Original Text', current_text))
        
        # Step 2: Convert to lowercase
        current_text = current_text.lower()
        if show_steps:
            steps.append(('Convert to Lowercase', current_text))
        
        # Step 3: Remove URLs (http, https, www)
        current_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_text)
        current_text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_text)
        if show_steps:
            steps.append(('Remove URLs', current_text))
        
        # Step 4: Remove email addresses
        current_text = re.sub(r'\S+@\S+\.\S+', '', current_text)
        if show_steps:
            steps.append(('Remove Email Addresses', current_text))
        
        # Step 5: Remove phone numbers (various formats)
        current_text = re.sub(r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '', current_text)
        current_text = re.sub(r'\+?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}', '', current_text)
        if show_steps:
            steps.append(('Remove Phone Numbers', current_text))
        
        # Step 6: Remove special characters (keep alphanumeric and spaces)
        current_text = re.sub(r'[^\w\s]', ' ', current_text)
        if show_steps:
            steps.append(('Remove Special Characters', current_text))
        
        # Step 7: Remove extra whitespace and normalize
        current_text = re.sub(r'\s+', ' ', current_text).strip()
        if show_steps:
            steps.append(('Remove Extra Whitespace', current_text))
        
        # Step 8: Final processed text
        processed_text = current_text
        if show_steps:
            steps.append(('Final Processed Text', processed_text))
        
        return {
            'original': text,
            'processed': processed_text,
            'steps': steps if show_steps else [],
            'character_reduction': len(text) - len(processed_text),
            'word_count_original': len(text.split()),
            'word_count_processed': len(processed_text.split()) if processed_text else 0
        }
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> Dict:
        """Get statistics about the preprocessing"""
        return {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'characters_removed': len(original_text) - len(processed_text),
            'reduction_percentage': ((len(original_text) - len(processed_text)) / len(original_text) * 100) if original_text else 0,
            'original_words': len(original_text.split()),
            'processed_words': len(processed_text.split()) if processed_text else 0,
            'words_removed': len(original_text.split()) - (len(processed_text.split()) if processed_text else 0)
        }
    
    def validate_preprocessing(self, text: str) -> Dict:
        """
        Validate that preprocessing produces reasonable results
        
        Returns:
            Dictionary with validation results and warnings
        """
        result = self.preprocess(text)
        warnings = []
        
        # Check if text becomes too short
        if len(result['processed']) < 10:
            warnings.append("Processed text is very short (< 10 characters)")
        
        # Check if too many characters were removed
        reduction_pct = self.get_preprocessing_stats(text, result['processed'])['reduction_percentage']
        if reduction_pct > 50:
            warnings.append(f"High character reduction ({reduction_pct:.1f}%) - may have removed important content")
        
        # Check if no words remain
        if result['processed'].strip() == "":
            warnings.append("No content remains after preprocessing")
        
        # Check for suspicious patterns that might indicate preprocessing issues
        if re.search(r'\d{4,}', result['processed']):
            warnings.append("Long number sequences detected - may need specific handling")
        
        return {
            'is_valid': len(warnings) == 0,
            'warnings': warnings,
            'processed_text': result['processed'],
            'stats': self.get_preprocessing_stats(text, result['processed'])
        }
    
    def preprocess_for_model(self, text: str, model_type: str = 'general') -> str:
        """
        Preprocess text specifically for different model types
        
        Args:
            text: Input text
            model_type: Type of model ('svm', 'lstm', 'bert', 'general')
            
        Returns:
            Preprocessed text optimized for the specific model
        """
        # Start with standard preprocessing
        result = self.preprocess(text)
        processed_text = result['processed']
        
        # Model-specific adjustments
        if model_type == 'svm':
            # SVM works well with standard preprocessing
            return processed_text
            
        elif model_type == 'lstm':
            # LSTM might benefit from preserving some structure
            # But we'll use standard preprocessing for consistency
            return processed_text
            
        elif model_type == 'bert':
            # BERT handles tokenization internally, so minimal preprocessing
            # Just remove URLs and emails, keep punctuation
            text_clean = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+\.\S+', '', text)
            return text_clean.strip()
            
        else:
            # General preprocessing
            return processed_text
    
    def display_preprocessing_steps(self, steps: List[Tuple[str, str]]) -> str:
        """
        Format preprocessing steps for display in Streamlit
        
        Args:
            steps: List of (step_name, step_text) tuples
            
        Returns:
            HTML formatted string for display
        """
        if not steps:
            return ""
        
        html = "<div style='font-family: monospace;'>"
        
        for i, (step_name, step_text) in enumerate(steps):
            html += f"<div style='margin-bottom: 10px;'>"
            html += f"<strong>{i+1}. {step_name}:</strong><br/>"
            
            # Truncate long text for display
            display_text = step_text[:200] + "..." if len(step_text) > 200 else step_text
            
            # Escape HTML characters
            display_text = display_text.replace('<', '&lt;').replace('>', '&gt;')
            
            html += f"<span style='background-color: #f0f0f0; padding: 5px; display: inline-block; width: 100%;'>{display_text}</span>"
            html += f"</div>"
        
        html += "</div>"
        return html
