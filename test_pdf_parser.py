import unittest
from unittest.mock import patch, MagicMock
import pdf_parser
import json

class TestPDFExtraction(unittest.TestCase):

    @patch('pdf_parser.PdfReader')
    @patch('pdf_parser.OpenAI')  # Mock the OpenAI class imported in pdf_parser
    @patch('pdf_parser.os.getenv')
    def test_llm_extraction_success(self, mock_getenv, mock_openai_cls, mock_reader):
        """Test that LLM extraction is called and returns formatted data when key exists."""
        
        # 1. Setup Mock API Key
        mock_getenv.return_value = "sk-fake-key"
        
        # 2. Setup Mock PDF Content
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Patient Report: Glucose 120, Age 55."
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance
        
        # 3. Setup Mock OpenAI Response
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"Glucose": 120, "Age": 55, "BMI": 28.5}\n```'
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # 4. Run Extraction
        result = pdf_parser.extract_data_from_pdf("dummy.pdf")
        
        # 5. Assertions
        mock_openai_cls.assert_called_with(api_key="sk-fake-key")
        self.assertEqual(result['Glucose'], 120)
        self.assertEqual(result['Age'], 55)
        self.assertEqual(result['BMI'], 28.5)
        print("\n[SUCCESS] OpenAI Extraction logic verified (Mocked).")

    @patch('pdf_parser.PdfReader')
    @patch('pdf_parser.os.getenv')
    def test_fallback_regex(self, mock_getenv, mock_reader):
        """Test fallback to regex if API Key is missing."""
        
        # 1. Simulate NO API Key
        mock_getenv.return_value = None
        
        # 2. Setup Mock PDF Content
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Lab Report\nGlucose Fasting: 99 mg/dL\nAge: 40 Years"
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance
        
        # 3. Run Extraction
        result = pdf_parser.extract_data_from_pdf("dummy.pdf")
        
        # 4. Assertions
        self.assertEqual(result.get('Glucose'), 99.0)
        self.assertEqual(result.get('Age'), 40.0)
        print("\n[SUCCESS] Regex Fallback logic verified.")

if __name__ == '__main__':
    unittest.main()
